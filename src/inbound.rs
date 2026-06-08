use anyhow::Result;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use reqwest::Client;

use crate::types::{HubMessage, PipelineState};

pub async fn run_inbound_listener(
    port: u16,
    worker_id: &str,
    pipeline: Arc<RwLock<PipelineState>>,
) -> Result<()> {
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Inbound listening started on {}", addr);
    let worker_id = worker_id.to_string();
    
    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                info!("Inbound connection from {} to worker {}", addr, worker_id);
                let pipeline_clone = pipeline.clone();
                let worker_id_clone = worker_id.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_inbound_connection(stream, pipeline_clone, &worker_id_clone).await {
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
    worker_id: &str,
) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    
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
                        
                        let response = HubMessage::InferenceResponse(crate::types::InferenceResponse {
                            id: req.id,
                            token: None,
                            hidden_states: None,
                            is_done: true,
                            text: Some(content),
                            prompt_tokens: 0,
                            completion_tokens: 0,
                        });
                        
                        let data = crate::protocol::encode_msg(&response);
                        stream.write_all(&data).await?;
                    }
                }
                Err(e) => {
                    error!("LLM request failed: {}", e);
                    let response = HubMessage::Error {
                        code: "LLM_ERROR".to_string(),
                        message: e.to_string(),
                    };
                    let data = crate::protocol::encode_msg(&response);
                    stream.write_all(&data).await?;
                }
            }
        }
        HubMessage::InferenceForward(fwd) => {
            info!("Received InferenceForward from {} to {}", fwd.from_worker, fwd.to_worker);
            let hidden_states: Vec<f32> = fwd.data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let response = HubMessage::InferenceResponse(crate::types::InferenceResponse {
                id: fwd.id,
                token: None,
                hidden_states: Some(hidden_states),
                is_done: false,
                text: None,
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            let data = crate::protocol::encode_msg(&response);
            stream.write_all(&data).await?;
        }
        HubMessage::HeartbeatForward { pipeline: pipeline_info } => {
            info!("Received HeartbeatForward for worker {}", worker_id);
            let pipeline_owned = pipeline_info.clone();
            let my_id = worker_id.to_string();
            
            if let Some(my_worker) = pipeline_owned.workers.iter().find(|w| w.worker_id == my_id) {
                // Send Heartbeat to hub if first or last worker
                let hub_addr = {
                    let pipeline_guard = pipeline.read().await;
                    pipeline_guard.hub_addr.clone()
                };
                
                if my_worker.is_first || my_worker.is_last {
                    info!("[self] {} worker - sending Heartbeat to hub", if my_worker.is_first { "first" } else { "last" });
                    let hb = crate::types::WorkerHeartbeat {
                        worker_id: my_id.clone(),
                        load: 0.0,
                        layer_offset: my_worker.layer_offset,
                        num_layers: my_worker.num_layers,
                        has_gpu: false,
                        vram_gb: 0.0,
                        active: true,
                        last_hop_connected: my_worker.last_hop.is_some(),
                        next_hop_connected: my_worker.next_hop.is_some(),
                    };
                    let response = HubMessage::Heartbeat(hb);
                    if let Ok(data) = crate::protocol::encode_msg(&response) {
                        if let Ok(mut hub_stream) = tokio::net::TcpStream::connect(&hub_addr).await {
                            hub_stream.write_all(&data).await.ok();
                            info!("[-> hub] Heartbeat sent for layers {}-{}", my_worker.layer_offset, my_worker.layer_offset + my_worker.num_layers);
                        } else {
                            warn!("Failed to connect to hub at {}", hub_addr);
                        }
                    }
                }
                
                // Forward to next hop if exists
                if let Some(ref hop) = my_worker.next_hop {
                    let addr = format!("{}:{}", hop.host, hop.port);
                    info!("[-> {}] Forwarding HeartbeatForward to next hop", hop.worker_id);
                    match tokio::net::TcpStream::connect(&addr).await {
                        Ok(mut forward_stream) => {
                            let data = crate::protocol::encode_msg(&HubMessage::HeartbeatForward { pipeline: pipeline_owned });
                            forward_stream.write_all(&data).await.ok();
                            info!("[-> {}] HeartbeatForward forwarded", hop.worker_id);
                        }
                        Err(e) => {
                            warn!("Failed to forward to next hop: {}", e);
                        }
                    }
                }
            }
        }
        _ => {
            info!("Unhandled inbound message type");
        }
    }

    Ok(())
}

pub async fn connect_to_next_hop(
    addr: &str,
) -> Result<TcpStream> {
    Ok(tokio::net::TcpStream::connect(addr).await?)
}