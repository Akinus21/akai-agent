mod build;
mod cli;
mod config;
mod gpu;
mod queue_client;
mod rpc;
mod service;
mod wireguard;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "akai_agent=info".into())
        )
        .init();
    cli::run().await
}