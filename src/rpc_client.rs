use anyhow::{bail, Result};
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::info;

const RPC_CONN_CAPS_SIZE: usize = 24;
const RPC_CMD_HELLO: u8 = 14;

pub struct RpcClient {
    addr: String,
}

impl RpcClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            addr: format!("{}:{}", host, port),
        }
    }

    async fn send_cmd(&self, cmd: u8, data: &[u8]) -> Result<Vec<u8>> {
        let mut stream = TcpStream::connect(&self.addr).await?;
        
        stream.write_all(&[cmd]).await?;
        stream.write_all(&(data.len() as u64).to_le_bytes()).await?;
        stream.write_all(data).await?;
        
        let mut size_buf = [0u8; 8];
        stream.read_exact(&mut size_buf).await?;
        let response_size = u64::from_le_bytes(size_buf) as usize;
        
        let mut response = vec![0u8; response_size];
        stream.read_exact(&mut response).await?;
        
        Ok(response)
    }

    pub async fn hello(&self) -> Result<(u8, u8, u8)> {
        let hello_req = vec![0u8; RPC_CONN_CAPS_SIZE];
        let resp = self.send_cmd(RPC_CMD_HELLO, &hello_req).await?;
        
        if resp.len() < 4 {
            bail!("hello response too short: {} bytes", resp.len());
        }
        
        let major = resp[0];
        let minor = resp[1];
        let patch = resp[2];
        info!("rpc-server: {}.{}.{}", major, minor, patch);
        
        Ok((major, minor, patch))
    }

    pub async fn init(&self, model_path: &str, layer_offset: usize, num_layers: usize) -> Result<()> {
        self.hello().await?;
        info!("rpc-server init() called with path={}, offset={}, layers={}", model_path, layer_offset, num_layers);
        Ok(())
    }

    pub async fn forward(&self, tokens: Vec<i64>, hidden_states: Option<Vec<f32>>) -> Result<(Vec<i64>, Vec<f32>)> {
        self.hello().await?;
        info!("rpc-server forward() called");
        Ok((tokens, hidden_states.unwrap_or_default()))
    }

    pub async fn generate(&self, tokens: Vec<i64>, max_new_tokens: usize, temperature: f32) -> Result<(Vec<i64>, String)> {
        self.hello().await?;
        info!("rpc-server generate() called");
        Ok((tokens, "placeholder".to_string()))
    }
}