use anyhow::Result;
use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;

use crate::vision::anchors::generate_anchors;

#[derive(Clone, Debug)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub landmarks: [(f32, f32); 5],
}

pub struct RetinaFace {
    session: Session,
}

impl RetinaFace {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<FaceBox>> {
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();

        let mut input_tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_img.get_pixel(x, y);
                input_tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 - 104.0;
                input_tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 - 117.0;
                input_tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 - 123.0;
            }
        }

        let input_value = Value::from_array(input_tensor.into_dyn())?;
        let outputs = self.session.run(ort::inputs![input_value])?;

        let bbox_reg_value = &outputs[0];
        let cls_value = &outputs[1];
        let ldmk_reg_value = &outputs[2];

        let (bbox_shape_raw, bbox_data_1d) = bbox_reg_value.try_extract_tensor::<f32>()?;
        let (_, cls_data_1d) = cls_value.try_extract_tensor::<f32>()?;
        let (_, ldmk_data_1d) = ldmk_reg_value.try_extract_tensor::<f32>()?;

        let num_anchors = bbox_shape_raw[1] as usize;
        let mut faces = Vec::new();

        let anchors = generate_anchors(height as usize, width as usize);
        let variance = [0.1, 0.2];

        // The number of generated anchors should match the model output shape.
        // If not, we will process up to the minimum to avoid panics.
        let limit = num_anchors.min(anchors.len());

        for i in 0..limit {
            let score = cls_data_1d[i * 2 + 1];
            if score > 0.5 {
                let anchor = &anchors[i];

                // Decode Bounding Box
                let dx1 = bbox_data_1d[i * 4];
                let dy1 = bbox_data_1d[i * 4 + 1];
                let dx2 = bbox_data_1d[i * 4 + 2];
                let dy2 = bbox_data_1d[i * 4 + 3];

                let cx = anchor.x + dx1 * variance[0] * anchor.w;
                let cy = anchor.y + dy1 * variance[0] * anchor.h;
                let w = anchor.w * (dx2 * variance[1]).exp();
                let h = anchor.h * (dy2 * variance[1]).exp();

                let x1 = cx - w / 2.0;
                let y1 = cy - h / 2.0;
                let x2 = cx + w / 2.0;
                let y2 = cy + h / 2.0;

                // Decode Landmarks
                let mut landmarks = [(0.0, 0.0); 5];
                for j in 0..5 {
                    let lmk_x = ldmk_data_1d[i * 10 + j * 2];
                    let lmk_y = ldmk_data_1d[i * 10 + j * 2 + 1];

                    let lx = anchor.x + lmk_x * variance[0] * anchor.w;
                    let ly = anchor.y + lmk_y * variance[0] * anchor.h;

                    landmarks[j] = (lx, ly);
                }

                faces.push(FaceBox {
                    x1,
                    y1,
                    x2,
                    y2,
                    score,
                    landmarks,
                });
            }
        }

        Ok(nms(faces, 0.4))
    }
}

pub(crate) fn nms(mut boxes: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(boxes[i].clone());

        for j in (i + 1)..boxes.len() {
            if suppressed[j] {
                continue;
            }

            let box_a = &boxes[i];
            let box_b = &boxes[j];

            let xx1 = box_a.x1.max(box_b.x1);
            let yy1 = box_a.y1.max(box_b.y1);
            let xx2 = box_a.x2.min(box_b.x2);
            let yy2 = box_a.y2.min(box_b.y2);

            let w = (xx2 - xx1).max(0.0);
            let h = (yy2 - yy1).max(0.0);
            let inter = w * h;

            let area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1);
            let area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1);
            let union = area_a + area_b - inter;

            if inter / union > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}
