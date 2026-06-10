mod auth;
mod build;
mod candle_llama;
mod candle_server;
mod candle_worker;
mod cli;
mod config;
mod gpu;
mod hub_worker;
mod inbound;
mod petals;
mod protocol;
mod queue_client;
mod rpc;
mod rpc_client;
mod service;
mod tunnel;
mod types;
mod wireguard;
mod worker;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "akai_agent=info".into())
        )
        .init();
    cli::run().await
}