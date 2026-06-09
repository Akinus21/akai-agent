use anyhow::{bail, Result};
use rmp_serde::Serializer;
use serde::{Deserialize, Serialize};
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

    fn serialize_req(req: &RpcRequest) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        let mut serializer = Serializer::new(&mut buf);
        req.serialize(&mut serializer).map_err(|e| anyhow::anyhow!("{}", e))?;
        buf.push(b'\n');
        Ok(buf)
    }

    pub async fn init(&self, model_path: &str, layer_offset: usize, num_layers: usize) -> Result<()> {
        let mut stream = TcpStream::connect(&self.addr).await?;
        
        // Send HELLO first (rpc-server expects this as first message)
        stream.write_all(b"hello\n").await?;
        
        let req = RpcRequest::Init {
            model_path: model_path.to_string(),
            layer_offset,
            num_layers,
        };
        
        let data = Self::serialize_req(&req)?;
        stream.write_all(&data).await?;
        
        let mut response_buf = vec![0u8; 65536];
        let n = stream.read(&mut response_buf).await?;
        
        if n == 0 {
            bail!("Connection closed during init");
        }
        
        let resp: RpcResponse = rmp_serde::from_slice(&response_buf[..n])
            .map_err(|e| anyhow::anyhow!("deserialization error: {}", e))?;
        
        match resp {
            RpcResponse::Done { .. } => {
                info!("rpc-server initialized successfully");
                Ok(())
            }
            RpcResponse::Error { message } => {
                bail!("rpc-server init error: {}", message);
            }
            _ => {
                bail!("Unexpected response type during init");
            }
        }
    }

    pub async fn forward(&self, tokens: Vec<i64>, hidden_states: Option<Vec<f32>>) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut stream = TcpStream::connect(&self.addr).await?;
        stream.write_all(b"hello\n").await?;
        
        let req = RpcRequest::Forward { tokens, hidden_states };
        let data = Self::serialize_req(&req)?;
        stream.write_all(&data).await?;
        
        let mut response_buf = vec![0u8; 65536];
        let n = stream.read(&mut response_buf).await?;
        
        if n == 0 {
            bail!("Connection closed during forward");
        }
        
        let resp: RpcResponse = rmp_serde::from_slice(&response_buf[..n])
            .map_err(|e| anyhow::anyhow!("deserialization error: {}", e))?;
        
        match resp {
            RpcResponse::HiddenStates { tokens, hidden_states } => {
                Ok((tokens, hidden_states))
            }
            RpcResponse::Error { message } => {
                bail!("rpc-server forward error: {}", message);
            }
            _ => {
                bail!("Unexpected response type during forward");
            }
        }
    }

    pub async fn generate(&self, tokens: Vec<i64>, max_new_tokens: usize, temperature: f32) -> Result<(Vec<i64>, String)> {
        let mut stream = TcpStream::connect(&self.addr).await?;
        stream.write_all(b"hello\n").await?;
        
        let req = RpcRequest::Generate {
            tokens,
            max_new_tokens,
            temperature,
        };
        let data = Self::serialize_req(&req)?;
        stream.write_all(&data).await?;
        
        let mut response_buf = vec![0u8; 65536];
        let n = stream.read(&mut response_buf).await?;
        
        if n == 0 {
            bail!("Connection closed during generate");
        }
        
        let resp: RpcResponse = rmp_serde::from_slice(&response_buf[..n])
            .map_err(|e| anyhow::anyhow!("deserialization error: {}", e))?;
        
        match resp {
            RpcResponse::Done { tokens, text } => {
                Ok((tokens, text))
            }
            RpcResponse::Error { message } => {
                bail!("rpc-server generate error: {}", message);
            }
            _ => {
                bail!("Unexpected response type during generate");
            }
        }
    }
}