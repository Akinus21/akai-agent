use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, TcpListener};
use tokio::sync::{Mutex, RwLock};
use tokio::time::Duration;
use tracing::{info, warn, error};
use reqwest::Client;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub layer_offset: usize,
    #[serde(default)]
    pub num_layers: usize,
    #[serde(default)]
    pub vram_gb: f32,
    #[serde(default)]
    pub has_gpu: bool,
    #[serde(default)]
    pub wg_ip: String,
    #[serde(default)]
    pub rpc_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HubMessage {
    #[serde(rename = "register")]
    Register(WorkerInfo),
    #[serde(rename = "heartbeat")]
    Heartbeat(WorkerHeartbeat),
    #[serde(rename = "heartbeat_response")]
    HeartbeatResponse(HeartbeatResponse),
    #[serde(rename = "heartbeat_forward")]
    HeartbeatForward { pipeline: PipelineInfo },
    #[serde(rename = "inference_request")]
    InferenceRequest(InferenceRequest),
    #[serde(rename = "inference_response")]
    InferenceResponse(InferenceResponse),
    #[serde(rename = "pipeline_info")]
    PipelineInfo(PipelineInfo),
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
    #[serde(default)]
    pub last_hop_connected: bool,
    #[serde(default)]
    pub next_hop_connected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub layer_offset: usize,
    pub num_layers: usize,
    pub reassign: bool,
    pub model_name: String,
    pub model_url: String,
    #[serde(default)]
    pub pipeline: Option<PipelineInfo>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopInfo {
    pub worker_id: String,
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineWorker {
    pub worker_id: String,
    pub layer_offset: usize,
    pub num_layers: usize,
    pub last_hop: Option<HopInfo>,
    pub next_hop: Option<HopInfo>,
    pub is_first: bool,
    pub is_last: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInfo {
    pub pipeline_id: String,
    pub workers: Vec<PipelineWorker>,
    pub model_name: String,
    pub model_url: String,
    pub num_layers: usize,
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
        name: config.worker_id.clone(),
        layer_offset: config.layer_offset,
        num_layers: config.num_layers,
        vram_gb: config.vram_gb,
        has_gpu: config.has_gpu,
        wg_ip: String::new(),
        rpc_port: 0,
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

pub struct PipelineState {
    pub my_worker_id: String,
    pub layer_offset: usize,
    pub num_layers: usize,
    pub last_hop: Option<HopInfo>,
    pub next_hop: Option<HopInfo>,
    pub last_hop_connected: bool,
    pub next_hop_connected: bool,
    pub is_first: bool,
    pub is_last: bool,
    pub next_hop_stream: Option<Arc<Mutex<TcpStream>>>,
}

impl PipelineState {
    pub fn new(worker_id: String) -> Self {
        Self {
            my_worker_id: worker_id,
            layer_offset: 0,
            num_layers: 0,
            last_hop: None,
            next_hop: None,
            last_hop_connected: false,
            next_hop_connected: false,
            is_first: false,
            is_last: false,
            next_hop_stream: None,
        }
    }
}

pub async fn run_hub_worker(config: HubWorkerConfig) -> Result<()> {
    info!("Akai-Net Hub Worker starting...");
    info!("  Hub: {}", config.hub_addr);
    info!("  Worker ID: {}", config.worker_id);
    info!("  GPU: {}, VRAM: {:.1} GB", config.has_gpu, config.vram_gb);
    info!("  RPC port: {}", config.rpc_port);

    let pipeline: Arc<RwLock<PipelineState>> = Arc::new(RwLock::new(PipelineState::new(config.worker_id.clone())));
    let rpc_child: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));

    // Spawn inbound connection listener for last_hop (weaker workers) to connect to us
    let inbound_port = config.rpc_port;
    let inbound_worker_id = config.worker_id.clone();
    let inbound_pipeline = pipeline.clone();
    tokio::spawn(async move {
        if let Err(e) = run_inbound_listener(inbound_port, &inbound_worker_id, inbound_pipeline).await {
            error!("Inbound listener error: {}", e);
        }
    });

    // Handle Ctrl+C gracefully
    let rpc_child_clone = rpc_child.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("Shutting down worker...");
        let opt_child = rpc_child_clone.lock().await.take();
        if let Some(mut child) = opt_child {
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
                    name: hostname::get()
                        .map(|h| h.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    layer_offset: 0,
                    num_layers: 0,
                    vram_gb: config.vram_gb,
                    has_gpu: config.has_gpu,
                    wg_ip: String::new(),
                    rpc_port: config.rpc_port,
                };
                let register = HubMessage::Register(worker_info);
                let data = serde_json::to_vec(&register)?;
                stream.write_all(&data).await?;
                info!("Registered with hub, waiting for layer assignment...");

                // Main communication loop
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
                        HubMessage::HeartbeatForward { pipeline: pipeline_info } => {
                            info!("Received heartbeat forward from hub, {} workers in pipeline", pipeline_info.workers.len());
                            
                            let pipeline_owned = pipeline_info.clone();
                            let my_id = &config.worker_id;
                            if let Some(my_worker) = pipeline_owned.workers.iter().find(|w| &w.worker_id == my_id) {
                                if let Some(ref hop) = my_worker.next_hop {
                                    let addr = format!("{}:{}", hop.host, hop.port);
                                    info!("Forwarding heartbeat to next hop: {} at {}", hop.worker_id, addr);
                                    match tokio::net::TcpStream::connect(&addr).await {
                                        Ok(mut forward_stream) => {
                                            let msg = HubMessage::HeartbeatForward { pipeline: pipeline_owned };
                                            let data = serde_json::to_vec(&msg).unwrap();
                                            forward_stream.write_all(&data).await.ok();
                                            info!("Heartbeat forwarded to {}", hop.worker_id);
                                        }
                                        Err(e) => {
                                            warn!("Failed to forward heartbeat to {}: {}", hop.worker_id, e);
                                        }
                                    }
                                } else {
                                    info!("This is the last worker in pipeline, sending heartbeat back to hub");
                                    let hub_port = std::env::var("HUB_PORT").unwrap_or_else(|_| "50051".to_string());
                                    let hub_addr = format!("{}:{}", config.hub_addr.split(':').next().unwrap_or("127.0.0.1"), hub_port);
                                    if let Ok(mut hub_stream) = tokio::net::TcpStream::connect(&hub_addr).await {
                                        let pipeline_guard = pipeline.read().await;
                                        let hb = WorkerHeartbeat {
                                            worker_id: config.worker_id.clone(),
                                            load: 0.0,
                                            layer_offset: pipeline_guard.layer_offset,
                                            num_layers: pipeline_guard.num_layers,
                                            has_gpu: config.has_gpu,
                                            vram_gb: config.vram_gb,
                                            active: true,
                                            last_hop_connected: my_worker.last_hop.is_some(),
                                            next_hop_connected: false,
                                        };
                                        let msg = HubMessage::Heartbeat(hb);
                                        let data = serde_json::to_vec(&msg).unwrap();
                                        hub_stream.write_all(&data).await.ok();
                                        info!("Sent cascade heartbeat back to hub");
                                    }
                                }
                            }
                        }
                        HubMessage::HeartbeatResponse(resp) => {
                            info!("Received heartbeat response: layers {} to {}, model={}, pipeline={}",
                                resp.layer_offset, resp.num_layers, resp.model_name, resp.pipeline.is_some());

                            let mut pipeline_guard = pipeline.write().await;
                            
                            // Update assignment if needed
                            if resp.reassign || pipeline_guard.num_layers == 0 {
                                pipeline_guard.layer_offset = resp.layer_offset;
                                pipeline_guard.num_layers = resp.num_layers;
                            }

                            // Update pipeline info
                            if let Some(pl) = resp.pipeline {
                                for w in &pl.workers {
                                    if w.worker_id == config.worker_id {
                                        pipeline_guard.last_hop = w.last_hop.clone();
                                        pipeline_guard.next_hop = w.next_hop.clone();
                                        pipeline_guard.is_first = w.is_first;
                                        pipeline_guard.is_last = w.is_last;
                                        break;
                                    }
                                }
                                
                                // Spawn rpc-server if we have assignment
                                if pipeline_guard.num_layers > 0 {
                                    info!("Spawning rpc-server for layers {} to {}...",
                                        pipeline_guard.layer_offset, pipeline_guard.layer_offset + pipeline_guard.num_layers);

                                    let rpc_path = crate::rpc::rpc_binary_path();
                                    if !rpc_path.exists() {
                                        info!("Downloading rpc-server...");
                                        crate::rpc::ensure_rpc_server().await
                                            .context("Failed to download rpc-server")?;
                                    }

                                    let child = crate::rpc::spawn_rpc_server(&rpc_path, config.rpc_port)?;
                                    rpc_child.lock().await.replace(child);
                                    info!("rpc-server started on port {}", config.rpc_port);
                                }
                            }
                        }
                        HubMessage::PipelineInfo(pl) => {
                            info!("Received pipeline update with {} workers", pl.workers.len());
                            let mut pipeline_guard = pipeline.write().await;
                            
                            let mut new_next_hop = None;
                            for w in &pl.workers {
                                if w.worker_id == config.worker_id {
                                    pipeline_guard.last_hop = w.last_hop.clone();
                                    pipeline_guard.next_hop = w.next_hop.clone();
                                    pipeline_guard.is_first = w.is_first;
                                    pipeline_guard.is_last = w.is_last;
                                    new_next_hop = w.next_hop.clone();
                                    info!("  My neighbors: last_hop={:?}, next_hop={:?}",
                                        pipeline_guard.last_hop, pipeline_guard.next_hop);
                                    break;
                                }
                            }
                            // Connect to next_hop if we have one and don't have a connection
                            if let Some(next_hop) = new_next_hop {
                                if pipeline_guard.next_hop_stream.is_none() {
                                    drop(pipeline_guard);
                                    let pipeline_clone = pipeline.clone();
                                    let next_hop_clone = next_hop.clone();
                                    tokio::spawn(async move {
                                        if let Err(e) = connect_to_next_hop(next_hop_clone, pipeline_clone).await {
                                            error!("Failed to connect to next_hop: {}", e);
                                        }
                                    });
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

async fn run_inbound_listener(
    port: u16,
    worker_id: &str,
    pipeline: Arc<RwLock<PipelineState>>,
) -> Result<()> {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    info!("Inbound listening started on 0.0.0.0:{}", port);
    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                info!("Inbound connection from {} to worker {}", addr, worker_id);
                let pipeline_clone = pipeline.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_connection(stream, pipeline_clone).await {
                        error!("Inbound handler error: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept inbound connection: {}", e);
            }
        }
    }
}

async fn handle_inbound_connection(
    mut stream: TcpStream,
    pipeline: Arc<RwLock<PipelineState>>,
) -> Result<()> {
    let mut buf = vec![0u8; 65536];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }

    let msg: HubMessage = match serde_json::from_slice(&buf[..n]) {
        Ok(m) => m,
        Err(e) => {
            error!("Failed to parse inbound message: {}", e);
            return Ok(());
        }
    };

    match msg {
        HubMessage::InferenceRequest(req) => {
            info!("Received inference request from neighbor: {} ({} tokens)", req.id, req.tokens.len());
            
            let client = Client::new();
            let llama_url = format!("http://127.0.0.1:{}/v1/chat/completions", 8080);
            
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
                    }
                }
                Err(e) => {
                    error!("llama-server request failed: {}", e);
                }
            }
        }
        _ => {}
    }
    
    Ok(())
}

async fn connect_to_next_hop(
    next_hop: HopInfo,
    pipeline: Arc<RwLock<PipelineState>>,
) -> Result<()> {
    let addr = format!("{}:{}", next_hop.host, next_hop.port);
    info!("Connecting to next_hop {} at {}", next_hop.worker_id, addr);
    
    match TcpStream::connect(&addr).await {
        Ok(stream) => {
            info!("Connected to next_hop {}", next_hop.worker_id);
            let mut pipeline_guard = pipeline.write().await;
            pipeline_guard.next_hop_stream = Some(Arc::new(Mutex::new(stream)));
            pipeline_guard.next_hop_connected = true;
            Ok(())
        }
        Err(e) => {
            error!("Failed to connect to next_hop {}: {}", next_hop.worker_id, e);
            let mut pipeline_guard = pipeline.write().await;
            pipeline_guard.next_hop_connected = false;
            Err(e.into())
        }
    }
}