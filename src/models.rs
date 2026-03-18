use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub async fn download_models(models_dir: &str) -> Result<()> {
    let retina_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/retinaface_mnet025_v2.onnx";
    let arcface_url = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx";

    let retina_path = format!("{}/retinaface.onnx", models_dir);
    let arcface_path = format!("{}/arcface.onnx", models_dir);

    download_file(retina_url, &retina_path).await?;
    download_file(arcface_url, &arcface_path).await?;

    Ok(())
}

async fn download_file(url: &str, path: &str) -> Result<()> {
    if Path::new(path).exists() {
        return Ok(());
    }
    println!("Downloading {}...", url);
    let response = reqwest::get(url).await?.bytes().await?;
    let mut file = File::create(path)?;
    file.write_all(&response)?;
    Ok(())
}
