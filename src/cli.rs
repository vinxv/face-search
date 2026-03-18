use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "eye", version = "0.1", about = "EYE: Rust Face Retrieval CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Init {
        #[arg(long, default_value = "./models")]
        models_dir: String,
    },
    Index {
        #[arg(long)]
        image: Option<String>,
        #[arg(long)]
        dir: Option<String>,
        #[arg(long, default_value = "./models")]
        models_dir: String,
    },
    Search {
        #[arg(long)]
        image: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long, default_value = "./models")]
        models_dir: String,
    },
}
