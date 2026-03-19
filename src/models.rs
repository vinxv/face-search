use anyhow::Result;
use std::fs::{self, File};
use std::io::{self, Cursor};
use std::path::Path;

pub async fn download_models(models_dir: &str) -> Result<()> {
    fs::create_dir_all(models_dir)?;

    // InsightFace Buffalo_L Model Pack (Contains SCRFD det_10g and ArcFace w600k_r50)
    let buffalo_l_url =
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip";
    let zip_path = format!("{}/buffalo_l.zip", models_dir);

    let det_path = format!("{}/det_10g.onnx", models_dir);
    let rec_path = format!("{}/w600k_r50.onnx", models_dir);

    if Path::new(&det_path).exists() && Path::new(&rec_path).exists() {
        println!("Models already downloaded.");
        return Ok(());
    }

    if !Path::new(&zip_path).exists() {
        println!("Downloading InsightFace model pack: buffalo_l.zip ...");
        let response = reqwest::get(buffalo_l_url).await?.bytes().await?;
        let mut file = File::create(&zip_path)?;
        io::copy(&mut Cursor::new(response), &mut file)?;
    }

    println!("Extracting models...");
    let file = File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => path.to_owned(),
            None => continue,
        };

        let file_name = outpath.file_name().unwrap_or_default().to_string_lossy();

        if file_name == "det_10g.onnx" || file_name == "w600k_r50.onnx" {
            let target_path = format!("{}/{}", models_dir, file_name);
            let mut outfile = File::create(&target_path)?;
            io::copy(&mut file, &mut outfile)?;
            println!("Extracted {}", file_name);
        }
    }

    Ok(())
}
