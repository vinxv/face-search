use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "eye", version = "0.1", about = "EYE: Rust Face Retrieval CLI")]
pub struct Cli {
    #[arg(long, default_value = "config.toml", global = true)]
    pub config: String,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Init,
    Index {
        #[arg(long)]
        image: Option<String>,
        #[arg(long)]
        dir: Option<String>,
    },
    Search {
        #[arg(long)]
        image: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
}
