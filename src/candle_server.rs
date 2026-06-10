use anyhow::{bail, Result};
use candle_core::Tensor;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn};

use crate::candle_llama::LayerLlama;

const CMD_HELLO: u8 = 14;
const CMD_INIT: u8 = 1;
const CMD_FORWARD: u8 = 2;
const CMD_GENERATE: u8 = 3;

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
        info!("Initializing Candle model: {} (layers {}-{})",
            model_path, self.layer_offset, self.layer_offset + self.num_layers);

        let llama = LayerLlama::load_with_layers(model_path, self.layer_offset, self.num_layers)?;

        let mut model = self.model.lock().await;
        *model = Some(llama);

        self.set_ready(true).await;
        info!("Candle model initialized successfully");

        Ok(())
    }

    pub async fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        let mut model_guard = self.model.lock().await;
        let model = model_guard.as_mut().ok_or_else(|| anyhow::anyhow!("model not initialized"))?;

        // Determine hidden size from model
        let hidden_size = model.hidden_size();
        let num_tokens = hidden_states.len() / hidden_size;

        // Reshape hidden states to 2D tensor
        let hidden = Tensor::from_slice(hidden_states, [num_tokens, hidden_size])?;

        // Run forward through our assigned layers
        let output = model.forward_hidden(&hidden, model.num_layers())?;

        // Convert back to Vec<f32>
        let output_vec = output.to_vec1()?;

        Ok(output_vec)
    }

    pub async fn generate(&self, hidden_states: &[f32], max_tokens: usize, temperature: f32) -> Result<(Vec<i64>, String)> {
        let mut model_guard = self.model.lock().await;
        let model = model_guard.as_mut().ok_or_else(|| anyhow::anyhow!("model not initialized"))?;

        let hidden_size = model.hidden_size();
        let num_tokens = hidden_states.len() / hidden_size;
        let hidden = Tensor::from_slice(hidden_states, [num_tokens, hidden_size])?;

        // Run forward through all our layers
        let output = model.forward_hidden(&hidden, model.num_layers())?;

        // Get logits from lm_head
        let logits = model.lm_head(&output)?;

        // Sample
        let (tokens, text) = model.sample(&logits, temperature, max_tokens)?;

        Ok((tokens, text))
    }
}

async fn read_message(stream: &mut TcpStream) -> Result<(u8, Vec<u8>)> {
    let mut cmd_buf = [0u8; 1];
    stream.read_exact(&mut cmd_buf).await?;
    let mut size_buf = [0u8; 8];
    stream.read_exact(&mut size_buf).await?;
    let size = u64::from_le_bytes(size_buf) as usize;
    let mut data = vec![0u8; size];
    stream.read_exact(&mut data).await?;
    Ok((cmd_buf[0], data))
}

async fn write_message(stream: &mut TcpStream, data: &[u8]) -> Result<()> {
    stream.write_all(&(data.len() as u64).to_le_bytes()).await?;
    stream.write_all(data).await?;
    Ok(())
}

async fn handle_hello(stream: &mut TcpStream) -> Result<()> {
    info!("HELLO received");
    let response = vec![0, 10, 0]; // major=0, minor=10, patch=0 (Candle 0.10.x)
    write_message(stream, &response).await?;
    Ok(())
}

async fn handle_init(stream: &mut TcpStream, data: &[u8], server: &CandleServer) -> Result<()> {
    info!("INIT received: {} bytes", data.len());

    // Parse INIT message: model_path\u0layer_offset\u0num_layers
    let msg = String::from_utf8_lossy(data);
    let parts: Vec<&str> = msg.split('\0').collect();

    if parts.len() < 3 {
        bail!("INIT message must have 3 null-terminated parts");
    }

    let model_path = parts[0];
    let layer_offset = parts[1].parse::<usize>().unwrap_or(0);
    let num_layers = parts[2].parse::<usize>().unwrap_or(32);

    server.init_model(model_path).await?;

    let response = vec![0]; // success
    write_message(stream, &response).await?;
    Ok(())
}

async fn handle_forward(stream: &mut TcpStream, data: &[u8], server: &CandleServer) -> Result<()> {
    info!("FORWARD received: {} bytes", data.len());

    // Data is hidden_states as f32 bytes
    let hidden_states: Vec<f32> = data
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    info!("Processing {} hidden state values", hidden_states.len());

    let output = server.forward(&hidden_states).await?;

    // Send back the processed hidden states
    let response: Vec<u8> = output.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    write_message(stream, &response).await?;

    Ok(())
}

async fn handle_generate(stream: &mut TcpStream, data: &[u8], server: &CandleServer) -> Result<()> {
    info!("GENERATE received: {} bytes", data.len());

    // Parse GENERATE: max_tokens(4) + temp(4) + hidden_states[f32]
    if data.len() < 8 {
        bail!("GENERATE message too short");
    }

    let max_tokens = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let temperature = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    // Remaining data is hidden states
    let hidden_data = &data[8..];
    let hidden_states: Vec<f32> = hidden_data
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let (tokens, text) = server.generate(&hidden_states, max_tokens, temperature).await?;

    // Send response: token_count(8) + tokens[] + text_len(8) + text
    let token_count = tokens.len() as u64;
    let mut response = token_count.to_le_bytes().to_vec();
    for token in &tokens {
        response.extend_from_slice(&token.to_le_bytes());
    }
    let text_bytes = text.as_bytes();
    response.extend_from_slice(&(text_bytes.len() as u64).to_le_bytes());
    response.extend_from_slice(text_bytes);

    write_message(stream, &response).await?;

    Ok(())
}

async fn handle_client(mut stream: TcpStream, server: Arc<CandleServer>) {
    info!("New client connection");

    loop {
        match read_message(&mut stream).await {
            Ok((cmd, data)) => {
                let result = match cmd {
                    CMD_HELLO => handle_hello(&mut stream).await,
                    CMD_INIT => handle_init(&mut stream, &data, &server).await,
                    CMD_FORWARD => handle_forward(&mut stream, &data, &server).await,
                    CMD_GENERATE => handle_generate(&mut stream, &data, &server).await,
                    _ => {
                        warn!("Unknown command: {}", cmd);
                        break;
                    }
                };

                if let Err(e) = result {
                    error!("Error handling command {}: {}", cmd, e);
                    break;
                }
            }
            Err(e) => {
                error!("Failed to read message: {}", e);
                break;
            }
        }
    }

    info!("Client disconnected");
}

pub async fn run_server(port: u16, layer_offset: usize, num_layers: usize) -> Result<()> {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Candle server listening on {}", addr);

    let server = Arc::new(CandleServer::new(layer_offset, num_layers));

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let server = server.clone();
                tokio::spawn(handle_client(stream, server));
            }
            Err(e) => {
                error!("Failed to accept connection: {}", e);
            }
        }
    }
}