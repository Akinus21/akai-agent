use anyhow::Result;
use std::process::Command;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};
use tokio::time::Duration;
use tracing::{error, info, warn};

use crate::config::data_dir;
use crate::protocol::{encode_msg, file_sha256};
use crate::types::{
    HubMessage, HubWorkerConfig, InferenceForward, InferenceResponse,
    PipelineState, WorkerHeartbeat, WorkerInfo,
};
use crate::{inbound, rpc, rpc_client};

pub fn notify(title: &str, body: &str) {
    info!("[notify] {}: {}", title, body);
    let _ = Command::new("notify-send")
        .args([title, body])
        .output();
}

fn setup_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        error!("PANIC in spawned task: {}", info);
    }));
}

pub async fn run_hub_worker(config: HubWorkerConfig) -> Result<()> {
    setup_panic_hook();
    info!("Akai-Net Hub Worker starting...");
    info!("  Hub: {}", config.hub_addr);
    info!("  Worker ID: {}", config.worker_id);
    info!("  GPU: {}, VRAM: {:.1} GB", config.has_gpu, config.vram_gb);
    info!("  RPC port: {}", config.rpc_port);
    info!("  LLM port: {}", config.llama_port);

    let pipeline: Arc<RwLock<PipelineState>> =
        Arc::new(RwLock::new(PipelineState::new(config.worker_id.clone(), config.hub_addr.clone())));
    let rpc_child: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));
    let llama_child: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));

    let inbound_port = config.rpc_port;
    let inbound_worker_id = config.worker_id.clone();
    let inbound_pipeline = pipeline.clone();
    tokio::spawn(async move {
        if let Err(e) = inbound::run_inbound_listener(inbound_port, &inbound_worker_id, inbound_pipeline).await {
            error!("Inbound listener error: {}", e);
        }
    });

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

    let mut reconnect_delay_secs = 5u64;
    let max_reconnect_delay = 300u64;
    loop {
        match TcpStream::connect(&config.hub_addr).await {
            Ok(stream) => {
                reconnect_delay_secs = 5;
                info!("Connected to hub at {}", config.hub_addr);

                let (reader, writer) = stream.into_split();
                let mut reader = reader;
                let writer = Arc::new(Mutex::new(writer));

                let worker_info = WorkerInfo {
                    id: config.worker_id.clone(),
                    name: hostname::get()
                        .map(|h| h.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    layer_offset: 0,
                    num_layers: 0,
                    vram_gb: config.vram_gb,
                    has_gpu: config.has_gpu,
                    wg_ip: config.wg_ip.clone(),
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
                                    info!("Hub connection closed (will retry in {}s, then increasing)", reconnect_delay_secs);
                                    reconnect_delay_secs = (reconnect_delay_secs * 2).min(max_reconnect_delay);
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

                                        handle_hub_message(
                                            &msg,
                                            config.clone(),
                                            pipeline.clone(),
                                            writer.clone(),
                                            rpc_child.clone(),
                                            llama_child.clone(),
                                        )
                                        .await;
                                    }
                                }
                                Err(e) => {
                                    error!("Read error: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to connect to hub: {} (retrying in {}s)", e, reconnect_delay_secs);
                tokio::time::sleep(Duration::from_secs(reconnect_delay_secs)).await;
                reconnect_delay_secs = (reconnect_delay_secs * 2).min(max_reconnect_delay);
            }
        }
    }
}

async fn handle_hub_message(
    msg: &HubMessage,
    config: HubWorkerConfig,
    pipeline: Arc<RwLock<PipelineState>>,
    writer: Arc<Mutex<tokio::net::tcp::OwnedWriteHalf>>,
    rpc_child: Arc<Mutex<Option<std::process::Child>>>,
    llama_child: Arc<Mutex<Option<std::process::Child>>>,
) {
    match msg {
        HubMessage::HeartbeatForward { pipeline: pipeline_info } => {
            info!(
                "[<- hub] HeartbeatForward: pipeline_id={}, {} workers, model={}",
                pipeline_info.pipeline_id,
                pipeline_info.workers.len(),
                pipeline_info.model_name
            );

            let pipeline_owned = pipeline_info.clone();
            let my_id = config.worker_id.clone();
            
            let (last_hop, next_hop, layer_offset, num_layers) = {
                let worker = pipeline_owned.workers.iter().find(|w| w.worker_id == my_id);
                match worker {
                    Some(w) => (w.last_hop.clone(), w.next_hop.clone(), w.layer_offset, w.num_layers),
                    None => return,
                }
            };
            
            // Derive position from hops - more robust than trusting hub's booleans
            let is_first = last_hop.is_none();
            let is_last = next_hop.is_none();
            
            {
                let mut pipeline_guard = pipeline.write().await;
                pipeline_guard.is_first = is_first;
                pipeline_guard.is_last = is_last;
                pipeline_guard.layer_offset = layer_offset;
                pipeline_guard.num_layers = num_layers;
                pipeline_guard.last_hop = last_hop.clone();
                pipeline_guard.next_hop = next_hop.clone();
            }
            
            let (llama_server_started, ready_for_inference) = {
                let pipeline_guard = pipeline.read().await;
                (pipeline_guard.llama_server_started, pipeline_guard.ready_for_inference)
            };
            
            let is_ready = llama_server_started && ready_for_inference;
            
            if !is_ready {
                info!("[self] not ready for inference yet (llama_server_started={}, ready={})", 
                    llama_server_started, ready_for_inference);
                if let Some(ref hop) = next_hop {
                    let addr = format!("{}:{}", hop.host, hop.port);
                    info!("[-> {}] Forwarding HeartbeatForward to next hop at {}", hop.worker_id, addr);
                    match tokio::net::TcpStream::connect(&addr).await {
                        Ok(mut forward_stream) => {
                            let msg = HubMessage::HeartbeatForward { pipeline: pipeline_owned };
                            let data = encode_msg(&msg);
                            forward_stream.write_all(&data).await.ok();
                            info!("[-> {}] HeartbeatForward sent", hop.worker_id);
                        }
                        Err(e) => {
                            warn!("[-> {}] HeartbeatForward FAILED: {}", hop.worker_id, e);
                        }
                    }
                }
                return;
            }
            
            if is_first {
                info!("[self] first worker in pipeline, sending Heartbeat back to hub");
                let pipeline_guard = pipeline.read().await;
                let (reported_offset, reported_num) = if let (Some(o), Some(n)) = (pipeline_guard.loaded_layer_offset, pipeline_guard.loaded_num_layers) {
                    (o, n)
                } else {
                    (pipeline_guard.layer_offset, pipeline_guard.num_layers)
                };
                let hb = WorkerHeartbeat {
                    worker_id: config.worker_id.clone(),
                    load: 0.0,
                    layer_offset: reported_offset,
                    num_layers: reported_num,
                    has_gpu: config.has_gpu,
                    vram_gb: config.vram_gb,
                    active: true,
                    last_hop_connected: last_hop.is_some(),
                    next_hop_connected: true,
                };
                let msg = HubMessage::Heartbeat(hb);
                let data = encode_msg(&msg);
                let mut w = writer.lock().await;
                w.write_all(&data).await.ok();
                info!(
                    "[-> hub] Heartbeat: layers {}-{}, active=true",
                    reported_offset,
                    reported_offset + reported_num
                );
            } else if is_last {
                info!("[self] last worker in pipeline, sending Heartbeat back to hub");
                let pipeline_guard = pipeline.read().await;
                let (reported_offset, reported_num) = if let (Some(o), Some(n)) = (pipeline_guard.loaded_layer_offset, pipeline_guard.loaded_num_layers) {
                    (o, n)
                } else {
                    (pipeline_guard.layer_offset, pipeline_guard.num_layers)
                };
                let hb = WorkerHeartbeat {
                    worker_id: config.worker_id.clone(),
                    load: 0.0,
                    layer_offset: reported_offset,
                    num_layers: reported_num,
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
                info!(
                    "[-> hub] Heartbeat: layers {}-{}, active=true, last_hop=true",
                    reported_offset,
                    reported_offset + reported_num
                );
            }

            if let Some(ref hop) = next_hop {
                let hop_worker_id = hop.worker_id.clone();
                let hop_host = hop.host.clone();
                let hop_port = hop.port;
                let addr = format!("{}:{}", hop_host, hop_port);
                info!(
                    "[-> {}] Forwarding HeartbeatForward to next hop at {}",
                    hop_worker_id, addr
                );
                match tokio::net::TcpStream::connect(&addr).await {
                    Ok(mut forward_stream) => {
                        let msg = HubMessage::HeartbeatForward {
                            pipeline: pipeline_owned,
                        };
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

        HubMessage::HeartbeatResponse(resp) => {
            info!(
                "[<- hub] HeartbeatResponse: layers {}-{}, model={}, pipeline={}",
                resp.layer_offset,
                resp.layer_offset + resp.num_layers,
                resp.model_name,
                resp.pipeline.is_some()
            );

            let mut pipeline_guard = pipeline.write().await;

            let layers_changed = resp.reassign && (resp.layer_offset != pipeline_guard.layer_offset || resp.num_layers != pipeline_guard.num_layers);
            
            if resp.reassign || pipeline_guard.num_layers == 0 {
                pipeline_guard.layer_offset = resp.layer_offset;
                pipeline_guard.num_layers = resp.num_layers;
                if layers_changed {
                    info!("Layer assignment changed, resetting setup state");
                    pipeline_guard.setup_started = false;
                    pipeline_guard.llama_server_started = false;
                    pipeline_guard.rpc_server_started = false;
                    pipeline_guard.loaded_layer_offset = None;
                    pipeline_guard.loaded_num_layers = None;
                    
                    // Kill running services so they restart with new layer params
                    let rpc_child = rpc_child.clone();
                    let llama_child = llama_child.clone();
                    tokio::spawn(async move {
                        if let Some(mut c) = rpc_child.lock().await.take() {
                            c.kill().ok();
                        }
                        if let Some(mut c) = llama_child.lock().await.take() {
                            c.kill().ok();
                        }
                    });
                }
            }

            let model_name_empty = resp.model_name.is_empty();
            let name_changed = !resp.model_name.is_empty() && resp.model_name != pipeline_guard.model_name;
            let url_changed = !resp.model_url.is_empty() && resp.model_url != pipeline_guard.model_url;
            let hash_changed = !resp.model_hash.is_empty() && resp.model_hash != pipeline_guard.model_hash;
            let model_changed = name_changed || url_changed || hash_changed;
            if model_changed {
                pipeline_guard.model_name.clone_from(&resp.model_name);
                pipeline_guard.model_url.clone_from(&resp.model_url);
            }
            if !resp.model_hash.is_empty() {
                pipeline_guard.model_hash.clone_from(&resp.model_hash);
            }

            if let Some(pl) = &resp.pipeline {
                for w in &pl.workers {
                    if w.worker_id == config.worker_id {
                        pipeline_guard.layer_offset = w.layer_offset;
                        pipeline_guard.num_layers = w.num_layers;
                        pipeline_guard.last_hop = w.last_hop.clone();
                        pipeline_guard.next_hop = w.next_hop.clone();
                        pipeline_guard.is_first = w.is_first;
                        pipeline_guard.is_last = w.is_last;
                        break;
                    }
                }

                if pipeline_guard.num_layers > 0 {
                    // Skip early rpc-server spawn - we spawn it AFTER model download

                    if pipeline_guard.num_layers > 0
                        && !pipeline_guard.model_url.is_empty()
                        && !pipeline_guard.setup_started
                    {
                        pipeline_guard.setup_started = true;
                        let model_url = pipeline_guard.model_url.clone();
                        let model_name = pipeline_guard.model_name.clone();
                        let has_gpu = config.has_gpu;
                        let llama_port = config.llama_port;
                        let layer_offset = pipeline_guard.layer_offset;
                        let num_layers = pipeline_guard.num_layers;
                        let pipeline_clone = pipeline.clone();
                        let model_hash_clone = pipeline_guard.model_hash.clone();
                        let llama_child_clone = llama_child.clone();

                        tokio::spawn(async move {
                            let model_path = data_dir().join("model.gguf");
                            let local_hash = file_sha256(&model_path);

                            let hash_matches = local_hash.as_ref().map_or(false, |h| {
                                !model_hash_clone.is_empty() && h == &model_hash_clone
                            });

                            let need_download = if !model_path.exists() {
                                true
                            } else if !model_hash_clone.is_empty() && !hash_matches {
                                true
                            } else {
                                false
                            };

                            if !need_download {
                                info!("Model file already exists and hash matches, skipping download");
                                if !pipeline_clone.read().await.llama_server_started {
                                    if let Ok(llama_path) = rpc::ensure_llama_server().await {
                                        let ngl = if has_gpu { -1 } else { 0 };
                                        match rpc::spawn_llama_server(
                                            &llama_path,
                                            &model_path.to_string_lossy(),
                                            ngl,
                                            llama_port,
                                            layer_offset,
                                            num_layers,
                                        ) {
                                            Ok(llama_cmd) => {
                                                info!("llama-server started on port {}", llama_port);
                                                llama_child_clone.lock().await.replace(llama_cmd);
                                                let mut pg = pipeline_clone.write().await;
                                                pg.llama_server_started = true;
                                                pg.ready_for_inference = true;
                                                notify(
                                                    "akai-agent",
                                                    &format!(
                                                        "{} ready for inference on port {}",
                                                        model_name, llama_port
                                                    ),
                                                );
                                            }
                                            Err(e) => {
                                                error!("Failed to spawn llama-server: {}", e);
                                            }
                                        }
                                    }
                                }
                                return;
                            }

                            {
                                let mut llama_guard = llama_child_clone.lock().await;
                                if let Some(mut old) = llama_guard.take() {
                                    info!("Model changed, stopping llama-server");
                                    old.kill().ok();
                                }
                            }
                            {
                                let mut pipeline_guard = pipeline_clone.write().await;
                                pipeline_guard.llama_server_started = false;
                            }

                            if model_path.exists() {
                                info!("Deleting old model file");
                                std::fs::remove_file(&model_path).ok();
                            }
                            info!("Downloading model from {}...", model_url);
                            notify(
                                "akai-agent",
                                &format!("Downloading model: {}", model_name),
                            );

                            let client = reqwest::Client::new();
                            match client
                                .get(&model_url)
                                .timeout(std::time::Duration::from_secs(600))
                                .send()
                                .await
                            {
                                Ok(resp) => {
                                    if resp.status().is_success() {
                                        let total = resp.content_length().unwrap_or(0);
                                        if total > 0 {
                                            info!(
                                                "Model size: {:.1} MB",
                                                total as f64 / 1_048_576.0
                                            );
                                        }
                                        let mut downloaded: u64 = 0;
                                        let mut last_pct: u64 = 0;
                                        if let Some(parent) = model_path.parent() {
                                            std::fs::create_dir_all(parent).ok();
                                        }
                                        let mut file =
                                            match std::fs::File::create(&model_path) {
                                                Ok(f) => f,
                                                Err(e) => {
                                                    error!(
                                                        "Failed to create model file: {}",
                                                        e
                                                    );
                                                    notify(
                                                        "akai-agent",
                                                        "Model download failed: cannot create file",
                                                    );
                                                    return;
                                                }
                                            };
                                        let mut stream = resp.bytes_stream();
                                        use futures_util::StreamExt;
                                        use std::io::Write;
                                        while let Some(chunk) = stream.next().await {
                                            let chunk = match chunk {
                                                Ok(c) => c,
                                                Err(e) => {
                                                    error!(
                                                        "Model download stream error: {}",
                                                        e
                                                    );
                                                    notify(
                                                        "akai-agent",
                                                        "Model download failed: stream error",
                                                    );
                                                    break;
                                                }
                                            };
                                            downloaded += chunk.len() as u64;
                                            if total > 0 {
                                                let pct = downloaded * 100 / total;
                                                if pct >= last_pct + 10 {
                                                    info!(
                                                        "Model download: {}% ({:.1}/{:.1} MB)",
                                                        pct,
                                                        downloaded as f64 / 1_048_576.0,
                                                        total as f64 / 1_048_576.0
                                                    );
                                                    last_pct = pct;
                                                }
                                            }
                                            if let Err(e) = file.write_all(&chunk) {
                                                error!("Model write error: {}", e);
                                                notify(
                                                    "akai-agent",
                                                    "Model download failed: write error",
                                                );
                                                break;
                                            }
                                        }
                                        if let Err(e) = file.flush() {
                                            error!("Model flush error: {}", e);
                                        }
                                        drop(file);
                                        info!(
                                            "Model downloaded to {:?} ({:.1} MB)",
                                            model_path,
                                            downloaded as f64 / 1_048_576.0
                                        );
                                        notify(
                                            "akai-agent",
                                            &format!(
                                                "Model downloaded ({:.1} MB)",
                                                downloaded as f64 / 1_048_576.0
                                            ),
                                        );
                                    } else {
                                        error!(
                                            "Model download failed with status: {}",
                                            resp.status()
                                        );
                                        notify(
                                            "akai-agent",
                                            &format!(
                                                "Model download failed: HTTP {}",
                                                resp.status()
                                            ),
                                        );
                                    }
                                }
                                Err(e) => {
                                    error!("Model download request failed: {}", e);
                                    notify(
                                        "akai-agent",
                                        &format!("Model download failed: {}", e),
                                    );
                                }
                            }

                            if model_path.exists() {
                                let mut pipeline_guard = pipeline_clone.write().await;
                                let layer_offset = pipeline_guard.layer_offset;
                                let num_layers = pipeline_guard.num_layers;
                                if !pipeline_guard.llama_server_started {
                                    match rpc::ensure_llama_server().await {
                                        Ok(llama_path) => {
                                            let ngl = if has_gpu { -1 } else { 0 };
                                            match rpc::spawn_llama_server(
                                                &llama_path,
                                                &model_path.to_string_lossy(),
                                                ngl,
                                                llama_port,
                                                layer_offset,
                                                num_layers,
                                            ) {
                                                Ok(llama_cmd) => {
                                                    info!(
                                                        "llama-server started on port {}",
                                                        llama_port
                                                    );
                                                    llama_child_clone
                                                        .lock()
                                                        .await
                                                        .replace(llama_cmd);
                                                    pipeline_guard.llama_server_started = true;
                                                    pipeline_guard.ready_for_inference = true;
                                                    drop(pipeline_guard);

                                                    // Spawn rpc-server AFTER model is ready
                                                    let rpc_port = config.rpc_port + 1;
                                                    let rpc_path = rpc::rpc_binary_path();
                                                    if !rpc_path.exists() {
                                                        rpc::ensure_rpc_server().await.ok();
                                                    }
                                                    let rc = rpc_child.clone();
                                                    match rpc::spawn_rpc_server(&rpc_path, rpc_port, layer_offset, num_layers) {
                                                        Ok(child) => {
                                                            rc.lock().await.replace(child);
                                                            let mp = model_path.to_string_lossy().to_string();
                                                            let pc = pipeline_clone.clone();
                                                            tokio::spawn(async move {
                                                                tokio::time::sleep(Duration::from_secs(2)).await;
                                                                let c = rpc_client::RpcClient::new("127.0.0.1", rpc_port);
                                                                if c.init(&mp, layer_offset, num_layers).await.is_ok() {
                                                                    let mut g = pc.write().await;
                                                                    g.rpc_server_started = true;
                                                                    g.ready_for_inference = true;
                                                                    g.loaded_layer_offset = Some(layer_offset);
                                                                    g.loaded_num_layers = Some(num_layers);
                                                                }
                                                            });
                                                        }
                                                        Err(e) => error!("Failed to spawn rpc-server: {}", e),
                                                    }

                                                    notify(
                                                        "akai-agent",
                                                        &format!(
                                                            "{} ready for inference on port {}",
                                                            model_name, llama_port
                                                        ),
                                                    );
                                                }
                                                Err(e) => {
                                                    error!(
                                                        "Failed to spawn llama-server: {}",
                                                        e
                                                    );
                                                    drop(pipeline_guard);
                                                    notify(
                                                        "akai-agent",
                                                        &format!(
                                                            "Failed to start llama-server: {}",
                                                            e
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            error!("Failed to ensure llama-server: {}", e);
                                            drop(pipeline_guard);
                                            notify(
                                                "akai-agent",
                                                &format!("Failed to find llama-server: {}", e),
                                            );
                                        }
                                    }
                                }
                            }
                            info!("Background setup task complete");
                        });
                    }
                }
            }
        }

        HubMessage::InferenceRequest(req) => {
            info!(
                "[<- hub] InferenceRequest: id={}, is_first={}, is_last={}, max_tokens={}",
                req.id, req.is_first, req.is_last, req.max_new_tokens
            );
            
            let pipeline_guard = pipeline.read().await;
            let is_first = pipeline_guard.is_first;
            let is_last = pipeline_guard.is_last;
            let next_hop = pipeline_guard.next_hop.clone();
            let layer_offset = pipeline_guard.layer_offset;
            let num_layers = pipeline_guard.num_layers;
            let rpc_server_started = pipeline_guard.rpc_server_started;
            drop(pipeline_guard);

            info!("[self] is_first={}, is_last={}, layers={}-{}, rpc_server_started={}", 
                is_first, is_last, layer_offset, layer_offset + num_layers, rpc_server_started);

            if !rpc_server_started {
                error!("[self] rpc-server not started, cannot process inference");
                let resp_msg = HubMessage::Error {
                    code: "NOT_READY".to_string(),
                    message: "rpc-server not started".to_string(),
                };
                let data = encode_msg(&resp_msg);
                let mut w = writer.lock().await;
                w.write_all(&data).await.ok();
                return;
            }

            let rpc_port = config.rpc_port + 1;
            let client = rpc_client::RpcClient::new("127.0.0.1", rpc_port);
            
            let prompt = req.prompt.clone().unwrap_or_default();
            let max_tokens = req.max_new_tokens;
            let temperature = req.temperature;

            // Unified logic: process layers, then forward or generate
            if let Some(ref hop) = next_hop {
                // Has next hop - run forward and forward to next worker
                info!("[self] has next_hop, running forward pass");
                match client.forward(vec![], None).await {
                    Ok((tokens, hidden_states)) => {
                        info!("[rpc] forward done, hidden_states len: {}", hidden_states.len());
                        
                        let hidden_bytes: Vec<u8> = hidden_states.iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        
                        info!("[-> {}] Forwarding hidden states ({} bytes) to next worker at {}:{}", 
                            hop.worker_id, hidden_bytes.len(), hop.host, hop.port);
                        let fwd = HubMessage::InferenceForward(InferenceForward {
                            id: req.id.clone(),
                            from_worker: config.worker_id.clone(),
                            to_worker: hop.worker_id.clone(),
                            data: hidden_bytes,
                            hub_addr: Some(config.hub_addr.clone()),
                        });
                        let data = encode_msg(&fwd);
                        match tokio::net::TcpStream::connect(format!("{}:{}", hop.host, hop.port)).await {
                            Ok(mut forward_stream) => {
                                forward_stream.write_all(&data).await.ok();
                                info!("[-> {}] Forwarded to next worker", hop.worker_id);
                            }
                            Err(e) => {
                                error!("[-> {}] Forward failed: {}", hop.worker_id, e);
                                let resp_msg = HubMessage::Error {
                                    code: "FORWARD_ERROR".to_string(),
                                    message: format!("Failed to forward to {}: {}", hop.worker_id, e),
                                };
                                let data = encode_msg(&resp_msg);
                                let mut w = writer.lock().await;
                                w.write_all(&data).await.ok();
                            }
                        }
                    }
                    Err(e) => {
                        error!("[rpc] forward failed: {}", e);
                        let resp_msg = HubMessage::Error {
                            code: "RPC_ERROR".to_string(),
                            message: e.to_string(),
                        };
                        let data = encode_msg(&resp_msg);
                        let mut w = writer.lock().await;
                        w.write_all(&data).await.ok();
                    }
                }
            } else {
                // No next hop - last worker, run forward then generate and return to hub
                info!("[self] last worker, running forward pass then generate");
                match client.forward(vec![], None).await {
                    Ok((tokens, hidden_states)) => {
                        info!("[rpc] forward done, hidden_states len: {}", hidden_states.len());
                        
                        match client.generate(tokens, max_tokens, temperature).await {
                            Ok((tokens, text)) => {
                                info!("[rpc] generate done, text length: {} chars", text.len());
                                let resp_msg = HubMessage::InferenceResponse(InferenceResponse {
                                    id: req.id.clone(),
                                    token: None,
                                    hidden_states: None,
                                    is_done: true,
                                    text: Some(text),
                                    prompt_tokens: 0,
                                    completion_tokens: tokens.len() as u64,
                                });
                                let data = encode_msg(&resp_msg);
                                let mut w = writer.lock().await;
                                w.write_all(&data).await.ok();
                            }
                            Err(e) => {
                                error!("[rpc] generate failed: {}", e);
                                let resp_msg = HubMessage::Error {
                                    code: "RPC_ERROR".to_string(),
                                    message: e.to_string(),
                                };
                                let data = encode_msg(&resp_msg);
                                let mut w = writer.lock().await;
                                w.write_all(&data).await.ok();
                            }
                        }
                    }
                    Err(e) => {
                        error!("[rpc] forward failed: {}", e);
                        let resp_msg = HubMessage::Error {
                            code: "RPC_ERROR".to_string(),
                            message: e.to_string(),
                        };
                        let data = encode_msg(&resp_msg);
                        let mut w = writer.lock().await;
                        w.write_all(&data).await.ok();
                    }
                }
            }
        }

        HubMessage::InferenceForward(fwd) => {
            info!(
                "[<- hub] InferenceForward: from={}, to={}",
                fwd.from_worker, fwd.to_worker
            );
            let pipeline_guard = pipeline.read().await;
            if let Some(ref hop) = pipeline_guard.next_hop {
                let addr = format!("{}:{}", hop.host, hop.port);
                info!("[-> {}] Forwarding to next hop at {}", hop.worker_id, addr);
                match tokio::net::TcpStream::connect(&addr).await {
                    Ok(mut forward_stream) => {
                        let data = encode_msg(msg);
                        forward_stream.write_all(&data).await.ok();
                    }
                    Err(e) => {
                        warn!("[-> {}] InferenceForward FAILED: {}", hop.worker_id, e);
                    }
                }
            }
        }

        HubMessage::Error { code, message } => {
            error!("[<- hub] Error: {} - {}", code, message);
        }

        _ => {
            warn!("[<- hub] Unhandled message type: {:?}", msg);
        }
    }
}