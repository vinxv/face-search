use anyhow::Result;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub models: ModelsConfig,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    pub uri: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelsConfig {
    pub base_dir: String,
    pub detection: DetectionConfig,
    pub recognition: RecognitionConfig,
}

#[derive(Debug, Deserialize)]
pub struct DetectionConfig {
    pub file: String,
    pub conf_threshold: f32,
    pub nms_threshold: f32,
}

#[derive(Debug, Deserialize)]
pub struct RecognitionConfig {
    pub file: String,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
