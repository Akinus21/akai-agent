use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn encode_msg(msg: &HubMessage) -> Vec<u8> {
    let mut data = serde_json::to_vec(msg).unwrap_or_default();
    data.push(b'\n');
    data
}
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
    #[serde(rename = "inference_request")]
    InferenceRequest(InferenceRequest),
    #[serde(rename = "inference_response")]
    InferenceResponse(InferenceResponse),
    #[serde(rename = "inference_forward")]
    InferenceForward(InferenceForward),
    #[serde(rename = "heartbeat")]
    Heartbeat(WorkerHeartbeat),
    #[serde(rename = "heartbeat_response")]
    HeartbeatResponse(HeartbeatResponse),
    #[serde(rename = "heartbeat_forward")]
    HeartbeatForward { pipeline: PipelineInfo },
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
    #[serde(default)]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: String,
    pub token: Option<i64>,
    pub hidden_states: Option<Vec<f32>>,
    pub is_done: bool,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceForward {
    pub id: String,
    pub from_worker: String,
    pub to_worker: String,
    pub data: Vec<u8>,
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
                let mut data = serde_json::to_vec(&register)?;
                data.push(b'\n');
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
                                            text: Some(content),
                                            prompt_tokens: 0,
                                            completion_tokens: 0,
                                        };
                                        let msg = HubMessage::InferenceResponse(resp);
                                        let data = encode_msg(&msg);
                                        stream.write_all(&data).await?;
                                    } else {
                                        error!("Failed to parse llama-server response");
                                        let resp = InferenceResponse {
                                            id: req.id,
                                            token: Some(0),
                                            hidden_states: None,
                                            is_done: true,
                                            text: None,
                                            prompt_tokens: 0,
                                            completion_tokens: 0,
                                        };
                                        let msg = HubMessage::InferenceResponse(resp);
                                        let data = encode_msg(&msg);
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
                                        text: None,
                                        prompt_tokens: 0,
                                        completion_tokens: 0,
                                    };
                                    let msg = HubMessage::InferenceResponse(resp);
                                    let data = encode_msg(&msg);
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
    pub llama_port: u16,
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
    pub model_name: String,
    pub model_url: String,
    pub llama_server_started: bool,
    pub rpc_server_started: bool,
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
            model_name: String::new(),
            model_url: String::new(),
            llama_server_started: false,
            rpc_server_started: false,
        }
    }
}

pub async fn run_hub_worker(config: HubWorkerConfig) -> Result<()> {
    info!("Akai-Net Hub Worker starting...");
    info!("  Hub: {}", config.hub_addr);
    info!("  Worker ID: {}", config.worker_id);
    info!("  GPU: {}, VRAM: {:.1} GB", config.has_gpu, config.vram_gb);
    info!("  RPC port: {}", config.rpc_port);
    info!("  LLM port: {}", config.llama_port);

    let pipeline: Arc<RwLock<PipelineState>> = Arc::new(RwLock::new(PipelineState::new(config.worker_id.clone())));
    let rpc_child: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));
    let llama_child: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));

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
    let llama_child_clone = llama_child.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("Shutting down worker...");
        let opt_child = rpc_child_clone.lock().await.take();
        if let Some(mut child) = opt_child {
            child.kill().ok();
        }
        let opt_child = llama_child_clone.lock().await.take();
        if let Some(mut child) = opt_child {
            child.kill().ok();
        }
        std::process::exit(0);
    });

    // Persistent connection to hub
    loop {
        match tokio::net::TcpStream::connect(&config.hub_addr).await {
            Ok(stream) => {
                info!("Connected to hub, registering...");

                let (reader, writer) = stream.into_split();
                let mut reader = reader;
                let writer = Arc::new(Mutex::new(writer));

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
                let mut data = serde_json::to_vec(&register)?;
                data.push(b'\n');
                {
                    let mut w = writer.lock().await;
                    w.write_all(&data).await?;
                }
                info!("Registered with hub, maintaining persistent connection...");

                let mut read_buf = Vec::new();
                let mut tmp = [0u8; 65536];
                loop {
                    tokio::select! {
                        n = reader.read(&mut tmp) => {
                            match n {
                                Ok(0) => {
                                    info!("Hub connection closed, reconnecting...");
                                    break;
                                }
                                Ok(n) => {
                                    read_buf.extend_from_slice(&tmp[..n]);

                                    while let Some(pos) = read_buf.iter().position(|&b| b == b'\n') {
                                        let line: Vec<u8> = read_buf.drain(..=pos).collect();
                                        let line = &line[..line.len() - 1];

                                        if line.is_empty() {
                                            continue;
                                        }

                                        let msg: HubMessage = match serde_json::from_slice(line) {
                                            Ok(m) => m,
                                            Err(e) => {
                                                error!("Failed to parse message: {}", e);
                                                continue;
                                            }
                                        };

                                    match msg {
                                        HubMessage::HeartbeatForward { pipeline: pipeline_info } => {
                                            info!("[<- hub] HeartbeatForward: pipeline_id={}, {} workers, model={}", 
                                                pipeline_info.pipeline_id, pipeline_info.workers.len(), pipeline_info.model_name);
                                            
                                            let pipeline_owned = pipeline_info.clone();
                                            let my_id = &config.worker_id;
                                            if let Some(my_worker) = pipeline_owned.workers.iter().find(|w| &w.worker_id == my_id) {
                                                if my_worker.is_first {
                                                    info!("[self] first worker in pipeline, sending Heartbeat back to hub");
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
                                                        next_hop_connected: true,
                                                    };
                                                    let msg = HubMessage::Heartbeat(hb);
                                                    let data = encode_msg(&msg);
                                                    let mut w = writer.lock().await;
                                                    w.write_all(&data).await.ok();
                                                    info!("[-> hub] Heartbeat: layers {}-{}, active=true", pipeline_guard.layer_offset, pipeline_guard.layer_offset + pipeline_guard.num_layers);
                                                } else if my_worker.is_last {
                                                    info!("[self] last worker in pipeline, sending Heartbeat back to hub");
                                                    let pipeline_guard = pipeline.read().await;
                                                    let hb = WorkerHeartbeat {
                                                        worker_id: config.worker_id.clone(),
                                                        load: 0.0,
                                                        layer_offset: pipeline_guard.layer_offset,
                                                        num_layers: pipeline_guard.num_layers,
                                                        has_gpu: config.has_gpu,
                                                        vram_gb: config.vram_gb,
                                                        active: true,
                                                        last_hop_connected: true,
                                                        next_hop_connected: false,
                                                    };
                                                    let msg = HubMessage::Heartbeat(hb);
                                                    let data = encode_msg(&msg);
                                                    let mut w = writer.lock().await;
                                                    w.write_all(&data).await.ok();
                                                    info!("[-> hub] Heartbeat: layers {}-{}, active=true, last_hop=true", pipeline_guard.layer_offset, pipeline_guard.layer_offset + pipeline_guard.num_layers);
                                                }
                                                
                                                if let Some(ref hop) = my_worker.next_hop {
                                                    let hop_worker_id = hop.worker_id.clone();
                                                    let hop_host = hop.host.clone();
                                                    let hop_port = hop.port;
                                                    let addr = format!("{}:{}", hop_host, hop_port);
                                                    info!("[-> {}] Forwarding HeartbeatForward to next hop at {}", hop_worker_id, addr);
                                                    match tokio::net::TcpStream::connect(&addr).await {
                                                        Ok(mut forward_stream) => {
                                                            let msg = HubMessage::HeartbeatForward { pipeline: pipeline_owned };
                                                            let data = encode_msg(&msg);
                                                            forward_stream.write_all(&data).await.ok();
                                                            info!("[-> {}] HeartbeatForward sent", hop_worker_id);
                                                        }
                                                        Err(e) => {
                                                            warn!("[-> {}] HeartbeatForward FAILED: {}", hop_worker_id, e);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        HubMessage::HeartbeatResponse(resp) => {
                                            info!("[<- hub] HeartbeatResponse: layers {}-{}, model={}, pipeline={}",
                                                resp.layer_offset, resp.layer_offset + resp.num_layers, resp.model_name, resp.pipeline.is_some());

                                            let mut pipeline_guard = pipeline.write().await;
                                            
                                            if resp.reassign || pipeline_guard.num_layers == 0 {
                                                pipeline_guard.layer_offset = resp.layer_offset;
                                                pipeline_guard.num_layers = resp.num_layers;
                                            }

                                            let model_changed = !resp.model_name.is_empty() && (resp.model_name != pipeline_guard.model_name || resp.model_url != pipeline_guard.model_url);
                                            if model_changed {
                                                pipeline_guard.model_name = resp.model_name.clone();
                                                pipeline_guard.model_url = resp.model_url.clone();
                                            }

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
                                                
                                                if pipeline_guard.num_layers > 0 {
                                                    // Spawn rpc-server if not already running
                                                    if !pipeline_guard.rpc_server_started {
                                                        info!("Spawning rpc-server for layers {} to {}...",
                                                            pipeline_guard.layer_offset, pipeline_guard.layer_offset + pipeline_guard.num_layers);

                                                        let rpc_path = crate::rpc::rpc_binary_path();
                                                        if !rpc_path.exists() {
                                                            info!("Downloading rpc-server...");
                                                            crate::rpc::ensure_rpc_server().await
                                                                .context("Failed to download rpc-server")?;
                                                        }

                                                        let child = crate::rpc::spawn_rpc_server(&rpc_path, config.rpc_port + 1)?;
                                                        rpc_child.lock().await.replace(child);
                                                        info!("rpc-server started on port {}", config.rpc_port + 1);
                                                        pipeline_guard.rpc_server_started = true;
                                                    }

                                                    // First worker: download model and start llama-server
                                                    if pipeline_guard.is_first && !pipeline_guard.model_url.is_empty() {
                                                        let model_path = crate::config::data_dir().join("model.gguf");

                                                        if model_changed || !model_path.exists() {
                                                            // Kill existing llama-server if running
                                                            if pipeline_guard.llama_server_started {
                                                                info!("Model changed, stopping llama-server");
                                                                if let Some(mut old) = llama_child.lock().await.take() {
                                                                    old.kill().ok();
                                                                }
                                                                pipeline_guard.llama_server_started = false;
                                                            }
                                                            // Delete old model and download new one
                                                            if model_path.exists() {
                                                                info!("Deleting old model file");
                                                                std::fs::remove_file(&model_path).ok();
                                                            }
                                                            info!("Downloading model from {}...", pipeline_guard.model_url);
                                                            let client = Client::new();
                                                            match client.get(&pipeline_guard.model_url)
                                                                .timeout(std::time::Duration::from_secs(600))
                                                                .send().await
                                                            {
                                                                Ok(resp) => {
                                                                    if resp.status().is_success() {
                                                                        let total = resp.content_length().unwrap_or(0);
                                                                        if total > 0 {
                                                                            info!("Model size: {:.1} MB", total as f64 / 1_048_576.0);
                                                                        }
                                                                        let mut downloaded: u64 = 0;
                                                                        let mut last_pct: u64 = 0;
                                                                        if let Some(parent) = model_path.parent() {
                                                                            std::fs::create_dir_all(parent).ok();
                                                                        }
                                                                        let mut file = std::fs::File::create(&model_path)?;
                                                                        let mut stream = resp.bytes_stream();
                                                                        use futures_util::StreamExt;
                                                                        while let Some(chunk) = stream.next().await {
                                                                            let chunk = match chunk {
                                                                                Ok(c) => c,
                                                                                Err(e) => {
                                                                                    error!("Model download stream error: {}", e);
                                                                                    break;
                                                                                }
                                                                            };
                                                                            downloaded += chunk.len() as u64;
                                                                            if total > 0 {
                                                                                let pct = downloaded * 100 / total;
                                                                                if pct >= last_pct + 10 {
                                                                                    info!("Model download: {}% ({:.1}/{:.1} MB)", 
                                                                                        pct, downloaded as f64 / 1_048_576.0, total as f64 / 1_048_576.0);
                                                                                    last_pct = pct;
                                                                                }
                                                                            }
                                                                            use std::io::Write;
                                                                            file.write_all(&chunk)?;
                                                                        }
                                                                        file.flush()?;
                                                                        drop(file);
                                                                        info!("Model downloaded to {:?} ({:.1} MB)", model_path, downloaded as f64 / 1_048_576.0);
                                                                    } else {
                                                                        error!("Model download failed with status: {}", resp.status());
                                                                    }
                                                                }
                                                                Err(e) => error!("Model download request failed: {}", e),
                                                            }
                                                        }
                                                        // Start llama-server if model exists and not already running
                                                        if model_path.exists() && !pipeline_guard.llama_server_started {
                                                            let llama_path = crate::rpc::llama_server_path();
                                                            crate::rpc::ensure_llama_server().await
                                                                .context("Failed to ensure llama-server")?;
                                                            let ngl = if config.has_gpu { -1 } else { 0 };
                                                            let llama_cmd = crate::rpc::spawn_llama_server(
                                                                &llama_path,
                                                                &model_path.to_string_lossy(),
                                                                ngl,
                                                                config.llama_port,
                                                            )?;
                                                            info!("llama-server started on port {}", config.llama_port);
                                                            llama_child.lock().await.replace(llama_cmd);
                                                            pipeline_guard.llama_server_started = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        HubMessage::InferenceRequest(req) => {
                                            info!("[<- hub] InferenceRequest: id={}, max_tokens={}", req.id, req.max_new_tokens);
                                            let prompt = req.prompt.unwrap_or_default();
                                            let max_tokens = req.max_new_tokens;
                                            let temperature = req.temperature;
                                            let llama_port = config.llama_port;

                                            let client = Client::new();
                                            let llm_url = format!("http://127.0.0.1:{}/v1/chat/completions", llama_port);

                                            let body = serde_json::json!({
                                                "model": "local",
                                                "messages": [{"role": "user", "content": prompt}],
                                                "max_tokens": max_tokens,
                                                "temperature": temperature,
                                                "stream": false
                                            });

                                            let resp_msg = match client.post(&llm_url)
                                                .json(&body)
                                                .timeout(std::time::Duration::from_secs(120))
                                                .send().await
                                            {
                                                Ok(resp) => {
                                                    match resp.json::<serde_json::Value>().await {
                                                        Ok(json) => {
                                                            let content = json["choices"][0]["message"]["content"]
                                                                .as_str().unwrap_or("").to_string();
                                                            let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
                                                            let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                                                            InferenceResponse {
                                                                id: req.id.clone(),
                                                                token: None,
                                                                hidden_states: None,
                                                                is_done: true,
                                                                text: Some(content),
                                                                prompt_tokens,
                                                                completion_tokens,
                                                            }
                                                        }
                                                        Err(e) => {
                                                            error!("Failed to parse LLM response: {}", e);
                                                            InferenceResponse {
                                                                id: req.id.clone(),
                                                                token: None,
                                                                hidden_states: None,
                                                                is_done: true,
                                                                text: Some(format!("Error parsing LLM response: {}", e)),
                                                                prompt_tokens: 0,
                                                                completion_tokens: 0,
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    error!("LLM request failed: {}", e);
                                                    InferenceResponse {
                                                        id: req.id.clone(),
                                                        token: None,
                                                        hidden_states: None,
                                                        is_done: true,
                                                        text: Some(format!("Error: LLM server not available - {}", e)),
                                                        prompt_tokens: 0,
                                                        completion_tokens: 0,
                                                    }
                                                }
                                            };

                                            let msg = HubMessage::InferenceResponse(resp_msg);
                                            let data = encode_msg(&msg);
                                            let mut w = writer.lock().await;
                                            w.write_all(&data).await?;
                                            info!("[-> hub] InferenceResponse: id={}, done={}", req.id, true);
                                        }
                                        HubMessage::InferenceForward(fwd) => {
                                            info!("[<- hub] InferenceForward: from={}, target={}, {} bytes",
                                                fwd.from_worker, fwd.to_worker, fwd.data.len());

                                            // Check if we are the target
                                            if fwd.to_worker == config.worker_id {
                                                // We are the target - process our layers
                                                // For now, if we have llama-server running, use it
                                                // If is_last, send InferenceResponse back to hub
                                                // If not is_last, forward to next worker
                                                let pipeline_guard = pipeline.read().await;
                                                if pipeline_guard.is_last {
                                                    // Last worker - run inference and return result to hub
                                                    drop(pipeline_guard);
                                                    let client = Client::new();
                                                    let llm_url = format!("http://127.0.0.1:{}/v1/chat/completions", config.llama_port);
                                                    let body = serde_json::json!({
                                                        "model": "local",
                                                        "messages": [{"role": "user", "content": "[forwarded inference data]"}],
                                                        "max_tokens": 128,
                                                        "temperature": 0.7,
                                                        "stream": false
                                                    });
                                                    match client.post(&llm_url).json(&body).timeout(std::time::Duration::from_secs(120)).send().await {
                                                        Ok(resp) => {
                                                            if let Ok(json) = resp.json::<serde_json::Value>().await {
                                                                let content = json["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
                                                                let pt = json["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
                                                                let ct = json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                                                                let resp_msg = InferenceResponse {
                                                                    id: fwd.id.clone(),
                                                                    token: None,
                                                                    hidden_states: None,
                                                                    is_done: true,
                                                                    text: Some(content),
                                                                    prompt_tokens: pt,
                                                                    completion_tokens: ct,
                                                                };
                                                                 let msg = HubMessage::InferenceResponse(resp_msg);
                                                                 let data = encode_msg(&msg);
                                                                 let mut w = writer.lock().await;
                                                                 w.write_all(&data).await?;
                                                                 info!("[-> hub] InferenceResponse: id={} (last worker)", fwd.id);
                                                            }
                                                        }
                                                        Err(e) => {
                                                            error!("LLM request failed in forward: {}", e);
                                                            let resp_msg = InferenceResponse {
                                                                id: fwd.id.clone(),
                                                                token: None,
                                                                hidden_states: None,
                                                                is_done: true,
                                                                text: Some(format!("Error: {}", e)),
                                                                prompt_tokens: 0,
                                                                completion_tokens: 0,
                                                            };
                                                            let msg = HubMessage::InferenceResponse(resp_msg);
                                                             let data = encode_msg(&msg);
                                                             let mut w = writer.lock().await;
                                                             w.write_all(&data).await?;
                                                        }
                                                    }
                                                } else {
                                                    // Not last worker - process our layers and forward to next
                                                    drop(pipeline_guard);
                                                    let pipeline_guard = pipeline.read().await;
                                                    if let Some(ref next_hop) = pipeline_guard.next_hop {
                                                        let forward = InferenceForward {
                                                            id: fwd.id.clone(),
                                                            from_worker: config.worker_id.clone(),
                                                            to_worker: next_hop.worker_id.clone(),
                                                            data: fwd.data.clone(),
                                                        };
                                                        let msg = HubMessage::InferenceForward(forward);
                                                        let data = encode_msg(&msg);
                                                        let mut w = writer.lock().await;
                                                        w.write_all(&data).await?;
                                                        info!("[-> hub] InferenceForward: -> next worker {}", next_hop.worker_id);
                                                    } else {
                                                        warn!("No next_hop configured, cannot forward inference");
                                                    }
                                                }
                                            }
                                        }
                                        HubMessage::Error { code, message } => {
                                            error!("Hub error {}: {}", code, message);
                                        }
                                        _ => {}
                                    }
                                } // end while let Some(pos)
                                }
                                Err(e) => {
                                    error!("[hub] Read error: {}", e);
                                }
                            }
                        }
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
    _pipeline: Arc<RwLock<PipelineState>>,
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
                            text: Some(content),
                            prompt_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                            completion_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0),
                        };
                        let msg = HubMessage::InferenceResponse(resp);
                        let data = encode_msg(&msg);
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