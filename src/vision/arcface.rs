use anyhow::Result;
use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;

pub struct ArcFace {
    session: Session,
}

impl ArcFace {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn extract(&mut self, aligned_face: &DynamicImage) -> Result<Vec<f32>> {
        let rgb_img = aligned_face.to_rgb8();
        if rgb_img.width() != 112 || rgb_img.height() != 112 {
            anyhow::bail!("Aligned face must be 112x112");
        }

        let mut input_tensor = Array4::<f32>::zeros((1, 3, 112, 112));
        for y in 0..112 {
            for x in 0..112 {
                let pixel = rgb_img.get_pixel(x, y);
                input_tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 127.5) - 1.0;
                input_tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 127.5) - 1.0;
                input_tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 127.5) - 1.0;
            }
        }

        let input_tensor_ort = Value::from_array(input_tensor.into_dyn())?;
        let outputs = self.session.run(ort::inputs![input_tensor_ort])?;

        let embedding_value = &outputs[0];
        let (_, embedding_data) = embedding_value.try_extract_tensor::<f32>()?;

        let mut embedding: Vec<f32> = embedding_data.to_vec();

        let mut sum_sq: f32 = 0.0;
        for val in &embedding {
            sum_sq += val * val;
        }
        let norm = sum_sq.sqrt().max(1e-10);

        for val in &mut embedding {
            *val /= norm;
        }

        Ok(embedding)
    }
}
