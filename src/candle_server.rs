use anyhow::{bail, Result};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn};

use crate::candle_llama::LayerLlama;

pub struct CandleServer {
    model: Arc<Mutex<Option<LayerLlama>>>,
    layer_offset: usize,
    num_layers: usize,
    ready: Arc<RwLock<bool>>,
}

impl CandleServer {
    pub fn new(layer_offset: usize, num_layers: usize) -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            layer_offset,
            num_layers,
            ready: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn set_ready(&self, ready: bool) {
        let mut r = self.ready.write().await;
        *r = ready;
    }

    pub async fn is_ready(&self) -> bool {
        *self.ready.read().await
    }

    pub async fn init_model(&self, model_path: &str) -> Result<()> {
        info!("Initializing GGUS model: {} (layers {}-{})",
            model_path, self.layer_offset, self.layer_offset + self.num_layers);

        let llama = LayerLlama::load_with_layers(model_path, self.layer_offset, self.num_layers)?;

        let mut model = self.model.lock().await;
        *model = Some(llama);

        self.set_ready(true).await;
        info!("GGUS model initialized successfully");

        Ok(())
    }

    pub async fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        let mut model_guard = self.model.lock().await;
        let model = model_guard.as_mut().ok_or_else(|| anyhow::anyhow!("model not initialized"))?;

        let num_tokens = 1;
        let output = model.forward_layers(hidden_states, num_tokens)?;
        Ok(output)
    }

    pub async fn generate(&self, hidden_states: &[f32], _max_tokens: usize, temperature: f32) -> Result<(Vec<i64>, String)> {
        let mut model_guard = self.model.lock().await;
        let model = model_guard.as_mut().ok_or_else(|| anyhow::anyhow!("model not initialized"))?;

        let num_tokens = 1;
        let after_layers = model.forward_layers(hidden_states, num_tokens)?;
        let logits = model.project(&after_layers, num_tokens)?;
        let (tokens, text) = model.sample(&logits, temperature)?;

        Ok((tokens, text))
    }
}

async fn handle_request(mut stream: TcpStream, server: Arc<CandleServer>) -> Result<()> {
    let mut buf = vec![0u8; 65536];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }

    let request_str = String::from_utf8_lossy(&buf[..n]);
    let lines: Vec<&str> = request_str.lines().collect();
    
    let method_path = lines.first().map(|s| *s).unwrap_or("");
    let parts: Vec<&str> = method_path.split_whitespace().collect();
    let method = parts.first().unwrap_or(&"");
    let path = parts.get(1).unwrap_or(&"");

    info!("HTTP {} {}", method, path);

    if *path == "/v1/chat/completions" && *method == "POST" {
        let body_start = request_str.find("\r\n\r\n").map(|s| s + 4).unwrap_or(0);
        let body = &request_str[body_start..];
        
        let body_json: serde_json::Value = match serde_json::from_str(body) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to parse request body: {}", e);
                send_error(&mut stream, 400, "Invalid JSON").await?;
                return Ok(());
            }
        };

        let model = body_json.get("model").and_then(|v| v.as_str()).unwrap_or("local-model");
        let messages = body_json.get("messages").and_then(|v| v.as_array());
        let temperature = body_json.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32;
        let max_tokens = body_json.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
        let stream_flag = body_json.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

        let user_message = messages
            .and_then(|msgs| {
                msgs.iter().rev().find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
            })
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        info!("Chat request: model={}, temp={}, max_tokens={}, stream={}", model, temperature, max_tokens, stream_flag);

        let hidden_size = {
            let model_guard = server.model.lock().await;
            if let Some(m) = model_guard.as_ref() {
                m.hidden_size()
            } else {
                1536
            }
        };

        let dummy_emb = vec![0.0f32; hidden_size];

        let (_tokens, text) = server.generate(&dummy_emb, max_tokens, temperature).await.unwrap_or_else(|_| {
            (vec![0], "Error generating response".to_string())
        });

        let response = if stream_flag {
            format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}]}}\r\n\r\ndata: [DONE]\r\n\r\n",
                text.replace("\"", "\\\"")
            )
        } else {
            serde_json::json!({
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2
                }
            }).to_string()
        };

        let response_len = response.len();
        let http_response = format!(
            "HTTP/1.1 200 OK\r\n\
            Content-Type: {}\r\n\
            Content-Length: {}\r\n\
            \r\n\
            {}",
            if stream_flag { "text/event-stream" } else { "application/json" },
            response_len,
            response
        );

        stream.write_all(http_response.as_bytes()).await?;
        Ok(())
    } else if *path == "/health" || *path == "/v1/models" {
        let response = serde_json::json!({
            "model": "local-model",
            "object": "list",
            "data": [{
                "id": "local-model",
                "object": "model",
                "created": 0,
                "owned_by": "local"
            }]
        }).to_string();

        let http_response = format!(
            "HTTP/1.1 200 OK\r\n\
            Content-Type: application/json\r\n\
            Content-Length: {}\r\n\
            \r\n\
            {}",
            response.len(),
            response
        );

        stream.write_all(http_response.as_bytes()).await?;
        Ok(())
    } else {
        send_error(&mut stream, 404, "Not found").await?;
        Ok(())
    }
}

async fn send_error(stream: &mut TcpStream, code: u16, message: &str) -> Result<()> {
    let response = format!(
        "HTTP/1.1 {} \r\n\
        Content-Type: application/json\r\n\
        Content-Length: {}\r\n\
        \r\n\
        {{\"error\": {{\"message\": \"{}\", \"type\": \"server_error\"}}}}",
        code,
        message.len() + 50,
        message
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

pub async fn run_server(port: u16, server: Arc<CandleServer>) -> Result<()> {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;
    info!("GGUS HTTP server listening on {}", addr);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let server = server.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_request(stream, server).await {
                        error!("Error handling request: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept connection: {}", e);
            }
        }
    }
}
