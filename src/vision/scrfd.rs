use anyhow::Result;
use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;

#[derive(Clone, Debug)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub landmarks: [(f32, f32); 5],
}

pub struct Scrfd {
    session: Session,
    conf_threshold: f32,
    nms_threshold: f32,
}

impl Scrfd {
    pub fn new(model_path: &str, conf_threshold: f32, nms_threshold: f32) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self {
            session,
            conf_threshold,
            nms_threshold,
        })
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<FaceBox>> {
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();

        // 1. Preprocess: Resize and Pad
        // SCRFD typically uses input dimensions that are multiples of 32.
        // For simplicity of inference in this generic pipeline, we pad to a dynamic size or
        // rely on the ONNX model's dynamic axes. InsightFace's det_10g is dynamic.
        let target_h = height.div_ceil(32) * 32;
        let target_w = width.div_ceil(32) * 32;

        let mut input_tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));

        // standard scaling for SCRFD: (pixel - 127.5) / 128.0
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_img.get_pixel(x, y);
                input_tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 128.0;
                input_tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 128.0;
                input_tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 128.0;
            }
        }

        let input_value = Value::from_array(input_tensor.into_dyn())?;
        let outputs = self.session.run(ort::inputs![input_value])?;

        // SCRFD outputs for stride 8, 16, 32. Order depends on the model export.
        // det_10g typically has 9 outputs:
        // [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32]
        let mut all_faces = Vec::new();

        let strides = [8, 16, 32];
        for (i, &stride) in strides.iter().enumerate() {
            let score_tensor = outputs[i * 3].try_extract_tensor::<f32>()?.1;
            let bbox_tensor = outputs[i * 3 + 1].try_extract_tensor::<f32>()?.1;
            let kps_tensor = outputs[i * 3 + 2].try_extract_tensor::<f32>()?.1;

            let feat_h = target_h as usize / stride;
            let feat_w = target_w as usize / stride;
            let num_anchors = 2;

            for y in 0..feat_h {
                for x in 0..feat_w {
                    for a in 0..num_anchors {
                        let idx = (y * feat_w + x) * num_anchors + a;

                        let score = score_tensor[idx];
                        if score > self.conf_threshold {
                            // Center point of the feature map mapped to image
                            let cx = (x as f32) * stride as f32;
                            let cy = (y as f32) * stride as f32;

                            // Decode BBox
                            // [l, t, r, b] offsets
                            let l = bbox_tensor[idx * 4] * stride as f32;
                            let t = bbox_tensor[idx * 4 + 1] * stride as f32;
                            let r = bbox_tensor[idx * 4 + 2] * stride as f32;
                            let b = bbox_tensor[idx * 4 + 3] * stride as f32;

                            let x1 = cx - l;
                            let y1 = cy - t;
                            let x2 = cx + r;
                            let y2 = cy + b;

                            // Decode Landmarks
                            let mut landmarks = [(0.0, 0.0); 5];
                            for j in 0..5 {
                                let lmk_x = cx + kps_tensor[idx * 10 + j * 2] * stride as f32;
                                let lmk_y = cy + kps_tensor[idx * 10 + j * 2 + 1] * stride as f32;
                                landmarks[j] = (lmk_x, lmk_y);
                            }

                            all_faces.push(FaceBox {
                                x1,
                                y1,
                                x2,
                                y2,
                                score,
                                landmarks,
                            });
                        }
                    }
                }
            }
        }

        Ok(nms(all_faces, self.nms_threshold))
    }
}

pub fn nms(mut boxes: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
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
