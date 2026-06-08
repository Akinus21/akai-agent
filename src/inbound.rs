use anyhow::Result;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tracing::{error, info};
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
            let response = HubMessage::InferenceResponse(crate::types::InferenceResponse {
                id: fwd.id,
                token: None,
                hidden_states: Some(fwd.data),
                is_done: false,
                text: None,
                prompt_tokens: 0,
                completion_tokens: 0,
            });
            let data = crate::protocol::encode_msg(&response);
            stream.write_all(&data).await?;
        }
        HubMessage::Heartbeat(hb) => {
            info!("Received Heartbeat from {} (active={})", hb.worker_id, hb.active);
            let response = HubMessage::HeartbeatResponse(crate::types::HeartbeatResponse {
                layer_offset: 0,
                num_layers: 0,
                reassign: false,
                model_name: String::new(),
                model_url: String::new(),
                model_hash: String::new(),
                pipeline: None,
            });
            let data = crate::protocol::encode_msg(&response);
            stream.write_all(&data).await?;
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