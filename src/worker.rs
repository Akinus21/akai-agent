use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};
use tracing::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub layer_offset: usize,
    pub num_layers: usize,
    pub vram_gb: f32,
    pub has_gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HubMessage {
    #[serde(rename = "register")]
    Register(WorkerInfo),
    #[serde(rename = "inference_request")]
    InferenceRequest(InferenceRequest),
    #[serde(rename = "inference_response")]
    InferenceResponse(InferenceResponse),
    #[serde(rename = "heartbeat")]
    Heartbeat {
        worker_id: String,
        load: f32,
        active: bool,
    },
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: String,
    pub tokens: Vec<i64>,
    pub is_first: bool,
    pub is_last: bool,
    pub max_new_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: String,
    pub token: Option<i64>,
    pub hidden_states: Option<Vec<f32>>,
    pub is_done: bool,
}

pub struct WorkerConfig {
    pub hub_addr: String,
    pub worker_id: String,
    pub has_gpu: bool,
    pub vram_gb: f32,
    pub layer_offset: usize,
    pub num_layers: usize,
    pub model_path: String,
}

pub async fn run_worker(config: WorkerConfig) -> Result<()> {
    info!("Akai-Net Worker starting...");
    info!("  Worker ID: {}", config.worker_id);
    info!("  Hub: {}", config.hub_addr);
    info!("  GPU: {}, VRAM: {:.1} GB", config.has_gpu, config.vram_gb);
    info!("  Layers: {} to {} ({})", config.layer_offset, config.layer_offset + config.num_layers, config.num_layers);

    let worker_info = WorkerInfo {
        id: config.worker_id.clone(),
        layer_offset: config.layer_offset,
        num_layers: config.num_layers,
        vram_gb: config.vram_gb,
        has_gpu: config.has_gpu,
    };

    loop {
        match TcpStream::connect(&config.hub_addr).await {
            Ok(mut stream) => {
                info!("Connected to hub at {}", config.hub_addr);

                let register = HubMessage::Register(worker_info.clone());
                let data = serde_json::to_vec(&register)?;
                stream.write_all(&data).await?;
                info!("Sent registration to hub");

                let mut buf = vec![0u8; 65536];
                while let Ok(n) = stream.read(&mut buf).await {
                    if n == 0 {
                        warn!("Connection closed by hub");
                        break;
                    }

                    let msg: HubMessage = match serde_json::from_slice(&buf[..n]) {
                        Ok(m) => m,
                        Err(e) => {
                            error!("Failed to parse message: {}", e);
                            continue;
                        }
                    };

                    match msg {
                        HubMessage::InferenceRequest(req) => {
                            info!("Received inference request {} ({} tokens)", req.id, req.tokens.len());
                            let resp = InferenceResponse {
                                id: req.id,
                                token: Some(0),
                                hidden_states: None,
                                is_done: true,
                            };
                            let msg = HubMessage::InferenceResponse(resp);
                            let data = serde_json::to_vec(&msg)?;
                            stream.write_all(&data).await?;
                        }
                        HubMessage::Heartbeat { worker_id, .. } => {
                            let resp = HubMessage::Heartbeat {
                                worker_id: worker_id.clone(),
                                load: 0.5,
                                active: true,
                            };
                            let data = serde_json::to_vec(&resp)?;
                            stream.write_all(&data).await?;
                        }
                        HubMessage::Error { code, message } => {
                            error!("Hub error {}: {}", code, message);
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                error!("Failed to connect to hub: {}", e);
            }
        }

        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
