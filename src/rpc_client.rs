use anyhow::{bail, Result};
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::info;

const RPC_CONN_CAPS_SIZE: usize = 24;
const RPC_CMD_HELLO: u8 = 14;
const RPC_CMD_INIT: u8 = 1;
const RPC_CMD_FORWARD: u8 = 2;
const RPC_CMD_GENERATE: u8 = 3;

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
        info!("candle-server: {}.{}.{}", major, minor, patch);

        Ok((major, minor, patch))
    }

    pub async fn init(&self, model_path: &str, layer_offset: usize, num_layers: usize) -> Result<()> {
        // Send hello first
        self.hello().await?;

        // Create INIT message: model_path\u0layer_offset\u0num_layers
        let mut data = model_path.as_bytes().to_vec();
        data.push(b'\0');
        data.extend_from_slice(&layer_offset.to_le_bytes());
        data.push(b'\0');
        data.extend_from_slice(&num_layers.to_le_bytes());

        let resp = self.send_cmd(RPC_CMD_INIT, &data).await?;

        if resp.is_empty() || resp[0] != 0 {
            bail!("init failed with response: {:?}", resp);
        }

        info!("candle-server init successful");
        Ok(())
    }

    pub async fn forward(&self, _tokens: Vec<i64>, hidden_states: Option<Vec<f32>>) -> Result<(Vec<i64>, Vec<f32>)> {
        // Send hello first
        self.hello().await?;

        // FORWARD message: hidden_states as f32 bytes
        let hidden = hidden_states.unwrap_or_default();
        let data: Vec<u8> = hidden.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let resp = self.send_cmd(RPC_CMD_FORWARD, &data).await?;

        // Response is processed hidden states as f32
        let output: Vec<f32> = resp
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok((_tokens, output))
    }

    pub async fn generate(&self, tokens: Vec<i64>, max_new_tokens: usize, temperature: f32) -> Result<(Vec<i64>, String)> {
        // Send hello first
        self.hello().await?;

        // For generate, we need hidden states which we don't have in this context
        // This is called on the last worker after forward
        // For now, return a placeholder - real implementation would need hidden states passed through

        // GENERATE message: max_tokens(4) + temp(4) - placeholder since we don't have hidden states
        let mut data = (max_new_tokens as u32).to_le_bytes().to_vec();
        data.extend_from_slice(&temperature.to_le_bytes());

        let resp = self.send_cmd(RPC_CMD_GENERATE, &data).await?;

        // Parse response: token_count(8) + tokens[] + text_len(8) + text
        if resp.len() < 16 {
            bail!("generate response too short: {} bytes", resp.len());
        }

        let mut pos = 0;
        let token_count = u64::from_le_bytes([resp[0], resp[1], resp[2], resp[3], resp[4], resp[5], resp[6], resp[7]]) as usize;
        pos += 8;

        let mut out_tokens = Vec::new();
        for _ in 0..token_count {
            if pos + 8 > resp.len() {
                break;
            }
            let token = i64::from_le_bytes([resp[pos], resp[pos+1], resp[pos+2], resp[pos+3],
                                           resp[pos+4], resp[pos+5], resp[pos+6], resp[pos+7]]);
            out_tokens.push(token);
            pos += 8;
        }

        let text_len = u64::from_le_bytes([resp[pos], resp[pos+1], resp[pos+2], resp[pos+3],
                                          resp[pos+4], resp[pos+5], resp[pos+6], resp[pos+7]]) as usize;
        pos += 8;

        let text = String::from_utf8_lossy(&resp[pos..pos+text_len]).to_string();

        Ok((out_tokens, text))
    }
}