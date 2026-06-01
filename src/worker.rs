use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::Duration;
use tracing::{info, warn, error};
use reqwest::Client;

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
    #[serde(rename = "heartbeat")]
    Heartbeat(WorkerHeartbeat),
    #[serde(rename = "inference_request")]
    InferenceRequest(InferenceRequest),
    #[serde(rename = "inference_response")]
    InferenceResponse(InferenceResponse),
    #[serde(rename = "heartbeat_response")]
    HeartbeatResponse(HeartbeatResponse),
    #[serde(rename = "error")]
    Error {
        code: String,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    pub worker_id: String,
    pub load: f32,
    pub layer_offset: usize,
    pub num_layers: usize,
    pub has_gpu: bool,
    pub vram_gb: f32,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub layer_offset: usize,
    pub num_layers: usize,
    pub reassign: bool,
    pub model_name: String,
    pub model_url: String,
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
    pub llama_server_path: Option<String>,
    pub n_gpu_layers: i32,
}

pub async fn run_worker(config: WorkerConfig) -> Result<()> {
    info!("Akai-Net Worker starting...");
    info!("  Worker ID: {}", config.worker_id);
    info!("  Hub: {}", config.hub_addr);
    info!("  GPU: {}, VRAM: {:.1} GB", config.has_gpu, config.vram_gb);
    info!("  Layers: {} to {} ({})", config.layer_offset, config.layer_offset + config.num_layers, config.num_layers);
    info!("  Model: {}", config.model_path);

    let worker_info = WorkerInfo {
        id: config.worker_id.clone(),
        layer_offset: config.layer_offset,
        num_layers: config.num_layers,
        vram_gb: config.vram_gb,
        has_gpu: config.has_gpu,
    };

    let llama_port = 8080u16;
    let n_gpu_layers = if config.has_gpu { config.n_gpu_layers } else { 0 };

    info!("Starting llama-server on port {}...", llama_port);
    let mut child = Command::new(config.llama_server_path.as_deref().unwrap_or("llama-server"))
        .args(&[
            "-m", &config.model_path,
            "-c", "4096",
            "-ngl", &n_gpu_layers.to_string(),
            "--port", &llama_port.to_string(),
            "--host", "127.0.0.1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    tokio::time::sleep(Duration::from_secs(5)).await;

    if let Some(status) = child.try_wait()? {
        if !status.success() {
            error!("llama-server exited with status: {}", status);
            anyhow::bail!("llama-server exited with status: {}", status);
        }
    }

    info!("llama-server started on 127.0.0.1:{}", llama_port);
    let client = Client::new();
    let llama_url = format!("http://127.0.0.1:{}/v1/chat/completions", llama_port);

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
                        HubMessage::HeartbeatResponse(resp) => {
                            info!("Heartbeat response: layers {} to {}, reassign={}",
                                resp.layer_offset, resp.num_layers, resp.reassign);
                        }
                        HubMessage::InferenceRequest(req) => {
                            info!("Received inference request {} ({} tokens)", req.id, req.tokens.len());

                            let prompt = format!("Tokens: {:?}", req.tokens);
                            let body = serde_json::json!({
                                "model": "local-model",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": req.max_new_tokens,
                                "temperature": req.temperature,
                                "stream": false
                            });

                            match client.post(&llama_url).json(&body).send().await {
                                Ok(resp) => {
                                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                                        let content = json["choices"][0]["message"]["content"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string();
                                        info!("Generated: {}", content);

                                        let resp = InferenceResponse {
                                            id: req.id,
                                            token: None,
                                            hidden_states: None,
                                            is_done: true,
                                        };
                                        let msg = HubMessage::InferenceResponse(resp);
                                        let data = serde_json::to_vec(&msg)?;
                                        stream.write_all(&data).await?;
                                    } else {
                                        error!("Failed to parse llama-server response");
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
                                }
                                Err(e) => {
                                    error!("llama-server request failed: {}", e);
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
                            }
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

pub struct HubWorkerConfig {
    pub hub_addr: String,
    pub worker_id: String,
    pub has_gpu: bool,
    pub vram_gb: f32,
    pub rpc_port: u16,
}

pub async fn run_hub_worker(config: HubWorkerConfig) -> Result<()> {
    info!("Akai-Net Hub Worker starting...");
    info!("  Hub: {}", config.hub_addr);
    info!("  Worker ID: {}", config.worker_id);
    info!("  GPU: {}, VRAM: {:.1} GB", config.has_gpu, config.vram_gb);
    info!("  RPC port: {}", config.rpc_port);

    let mut layer_offset: usize = 0;
    let mut num_layers: usize = 0;
    let mut rpc_child: Option<std::process::Child> = None;
    let mut assigned = false;

    // Handle Ctrl+C gracefully
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("Shutting down worker...");
        if let Some(child) = rpc_child.take() {
            child.kill().ok();
        }
        std::process::exit(0);
    });

    loop {
        match tokio::net::TcpStream::connect(&config.hub_addr).await {
            Ok(mut stream) => {
                info!("Connected to hub");

                // Register with hub
                let worker_info = WorkerInfo {
                    id: config.worker_id.clone(),
                    layer_offset: 0,
                    num_layers: 0,
                    vram_gb: config.vram_gb,
                    has_gpu: config.has_gpu,
                };
                let register = HubMessage::Register(worker_info);
                let data = serde_json::to_vec(&register)?;
                stream.write_all(&data).await?;
                info!("Registered with hub, waiting for layer assignment...");

                // Wait for heartbeat response or other messages
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
                        HubMessage::HeartbeatResponse(resp) => {
                            info!("Received heartbeat response: layers {} to {}, model={}",
                                resp.layer_offset, resp.num_layers, resp.model_name);
                            if resp.reassign || !assigned {
                                layer_offset = resp.layer_offset;
                                num_layers = resp.num_layers;
                                assigned = true;
                            }
                        }
                        HubMessage::Error { code, message } => {
                            error!("Hub error {}: {}", code, message);
                        }
                        _ => {}
                    }

                    // If we got our assignment, spawn rpc-server
                    if assigned && rpc_child.is_none() {
                        info!("Spawning rpc-server for layers {} to {}...", layer_offset, layer_offset + num_layers);

                        let rpc_path = crate::rpc::rpc_binary_path();
                        if !rpc_path.exists() {
                            info!("Downloading rpc-server...");
                            crate::rpc::ensure_rpc_server().await
                                .context("Failed to download rpc-server")?;
                        }

                        let child = crate::rpc::spawn_rpc_server(&rpc_path, config.rpc_port)?;
                        rpc_child = Some(child);
                        info!("rpc-server started on port {}", config.rpc_port);
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