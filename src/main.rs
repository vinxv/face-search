mod cli;
mod db;
mod models;
mod vision;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use image::io::Reader as ImageReader;
use walkdir::WalkDir;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { models_dir } => {
            println!("Initializing models in {}", models_dir);
            std::fs::create_dir_all(&models_dir)?;
            models::download_models(&models_dir).await?;
            println!("Initialization complete.");
        }
        Commands::Index {
            image,
            dir,
            models_dir,
        } => {
            let retina_path = format!("{}/retinaface.onnx", models_dir);
            let arcface_path = format!("{}/arcface.onnx", models_dir);
            let mut retinaface = vision::retinaface::RetinaFace::new(&retina_path)?;
            let mut arcface = vision::arcface::ArcFace::new(&arcface_path)?;

            let db = db::Database::new("./lance_db").await?;
            db.create_table().await?;

            let mut paths = Vec::new();
            if let Some(p) = image {
                paths.push(p);
            }
            if let Some(d) = dir {
                for entry in WalkDir::new(d).into_iter().filter_map(|e| e.ok()) {
                    if entry.file_type().is_file() {
                        if let Some(ext) = entry.path().extension() {
                            let ext_str = ext.to_string_lossy().to_lowercase();
                            if ext_str == "jpg" || ext_str == "jpeg" || ext_str == "png" {
                                paths.push(entry.path().to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }

            for path in paths {
                println!("Processing {}", path);
                let img = match ImageReader::open(&path) {
                    Ok(r) => match r.decode() {
                        Ok(i) => i,
                        Err(e) => {
                            println!("Failed to decode {}: {}", path, e);
                            continue;
                        }
                    },
                    Err(e) => {
                        println!("Failed to open {}: {}", path, e);
                        continue;
                    }
                };

                let faces = match retinaface.detect(&img) {
                    Ok(f) => f,
                    Err(e) => {
                        println!("Detection failed {}: {}", path, e);
                        continue;
                    }
                };

                for face in faces {
                    let aligned = vision::alignment::align_face(&img, &face.landmarks)?;
                    let embedding = arcface.extract(&aligned)?;
                    db.insert(&path, &face, &embedding).await?;
                }
            }
        }
        Commands::Search {
            image,
            top_k,
            models_dir,
        } => {
            let retina_path = format!("{}/retinaface.onnx", models_dir);
            let arcface_path = format!("{}/arcface.onnx", models_dir);
            let mut retinaface = vision::retinaface::RetinaFace::new(&retina_path)?;
            let mut arcface = vision::arcface::ArcFace::new(&arcface_path)?;

            let db = db::Database::new("./lance_db").await?;

            let img = ImageReader::open(&image)?.decode()?;
            let mut faces = retinaface.detect(&img)?;

            if faces.is_empty() {
                println!("No face found in query image.");
                return Ok(());
            }

            // Pick largest and most confident
            faces.sort_by(|a, b| {
                let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
                let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
                area_b.partial_cmp(&area_a).unwrap()
            });
            let main_face = &faces[0];

            let aligned = vision::alignment::align_face(&img, &main_face.landmarks)?;
            let embedding = arcface.extract(&aligned)?;

            let results = db.search(&embedding, top_k).await?;
            println!("Search Results:");
            for (path, score) in results {
                println!("{:.4} - {}", score, path);
            }
        }
    }
    Ok(())
}
