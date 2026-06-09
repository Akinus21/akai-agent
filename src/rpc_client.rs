use anyhow::{bail, Result};
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcRequest {
    Init {
        model_path: String,
        layer_offset: usize,
        num_layers: usize,
    },
    Forward {
        tokens: Vec<i64>,
        hidden_states: Option<Vec<f32>>,
    },
    Generate {
        tokens: Vec<i64>,
        max_new_tokens: usize,
        temperature: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcResponse {
    HiddenStates {
        tokens: Vec<i64>,
        hidden_states: Vec<f32>,
    },
    Done {
        tokens: Vec<i64>,
        text: String,
    },
    Error {
        message: String,
    },
}

pub struct RpcClient {
    addr: String,
}

impl RpcClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            addr: format!("{}:{}", host, port),
        }
    }

    pub async fn init(&self, model_path: &str, layer_offset: usize, num_layers: usize) -> Result<()> {
        let mut stream = TcpStream::connect(&self.addr).await?;
        
        let req = RpcRequest::Init {
            model_path: model_path.to_string(),
            layer_offset,
            num_layers,
        };
        
        let mut buf = Vec::new();
        req.serialize(&mut buf).unwrap();
        buf.push(b'\n');
        
        stream.write_all(&buf).await?;
        
        let mut response_buf = vec![0u8; 65536];
        let n = stream.read(&mut response_buf).await?;
        
        if n == 0 {
            bail!("Connection closed during init");
        }
        
        let mut de = Deserializer::new(&response_buf[..n]);
        match Deserialize::deserialize(&mut de) {
            Ok(RpcResponse::Done { .. }) => {
                info!("rpc-server initialized successfully");
                Ok(())
            }
            Ok(RpcResponse::Error { message }) => {
                bail!("rpc-server init error: {}", message);
            }
            Err(e) => {
                bail!("Failed to parse init response: {}", e);
            }
        }
    }

    pub async fn forward(&self, tokens: Vec<i64>, hidden_states: Option<Vec<f32>>) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut stream = TcpStream::connect(&self.addr).await?;
        stream.set_read_timeout(Some(Duration::from_secs(120)))?;
        
        let req = RpcRequest::Forward { tokens, hidden_states };
        
        let mut buf = Vec::new();
        req.serialize(&mut buf).unwrap();
        buf.push(b'\n');
        
        stream.write_all(&buf).await?;
        
        let mut response_buf = vec![0u8; 65536];
        let n = stream.read(&mut response_buf).await?;
        
        if n == 0 {
            bail!("Connection closed during forward");
        }
        
        let mut de = Deserializer::new(&response_buf[..n]);
        match Deserialize::deserialize(&mut de) {
            Ok(RpcResponse::HiddenStates { tokens, hidden_states }) => {
                Ok((tokens, hidden_states))
            }
            Ok(RpcResponse::Error { message }) => {
                bail!("rpc-server forward error: {}", message);
            }
            Ok(RpcResponse::Done { tokens, text: _ }) => {
                bail!("Unexpected Done response during forward");
            }
            Err(e) => {
                bail!("Failed to parse forward response: {}", e);
            }
        }
    }

    pub async fn generate(&self, tokens: Vec<i64>, max_new_tokens: usize, temperature: f32) -> Result<(Vec<i64>, String)> {
        let mut stream = TcpStream::connect(&self.addr).await?;
        stream.set_read_timeout(Some(Duration::from_secs(120)))?;
        
        let req = RpcRequest::Generate {
            tokens,
            max_new_tokens,
            temperature,
        };
        
        let mut buf = Vec::new();
        req.serialize(&mut buf).unwrap();
        buf.push(b'\n');
        
        stream.write_all(&buf).await?;
        
        let mut response_buf = vec![0u8; 65536];
        let n = stream.read(&mut response_buf).await?;
        
        if n == 0 {
            bail!("Connection closed during generate");
        }
        
        let mut de = Deserializer::new(&response_buf[..n]);
        match Deserialize::deserialize(&mut de) {
            Ok(RpcResponse::Done { tokens, text }) => {
                Ok((tokens, text))
            }
            Ok(RpcResponse::Error { message }) => {
                bail!("rpc-server generate error: {}", message);
            }
            Ok(RpcResponse::HiddenStates { .. }) => {
                bail!("Unexpected HiddenStates response during generate");
            }
            Err(e) => {
                bail!("Failed to parse generate response: {}", e);
            }
        }
    }
}