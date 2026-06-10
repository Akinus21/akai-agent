use anyhow::Result;
use std::sync::Arc;
use tracing::{error, info};

use crate::candle_server::{run_server, CandleServer};

pub struct CandleWorker {
    _join_handle: tokio::task::JoinHandle<()>,
}

impl CandleWorker {
    pub async fn start(
        port: u16,
        model_path: String,
        layer_offset: usize,
        num_layers: usize,
    ) -> Result<Self> {
        info!("Starting Candle server: port={}, layers={}-{}", port, layer_offset, num_layers);

        let server = Arc::new(CandleServer::new(layer_offset, num_layers));

        // Initialize the model first
        server.init_model(&model_path).await?;

        // Spawn the server task
        let server_clone = server;
        let join_handle = tokio::spawn(async move {
            if let Err(e) = run_server(port, layer_offset, num_layers).await {
                error!("Candle server error: {}", e);
            }
        });

        Ok(Self {
            _join_handle: join_handle,
        })
    }
}