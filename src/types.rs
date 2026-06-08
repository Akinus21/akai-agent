use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::Mutex;

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
    pub model_hash: String,
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

#[derive(Clone)]
pub struct HubWorkerConfig {
    pub hub_addr: String,
    pub worker_id: String,
    pub has_gpu: bool,
    pub vram_gb: f32,
    pub rpc_port: u16,
    pub llama_port: u16,
    pub wg_ip: String,
}

pub struct PipelineState {
    pub my_worker_id: String,
    pub hub_addr: String,
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
    pub model_hash: String,
    pub llama_server_started: bool,
    pub rpc_server_started: bool,
    pub setup_started: bool,
}

impl PipelineState {
    pub fn new(worker_id: String, hub_addr: String) -> Self {
        Self {
            my_worker_id: worker_id,
            hub_addr,
            layer_offset: 0,
            num_layers: 0,
            last_hop: None,
            next_hop: None,
            last_hop_connected: false,
            next_hop_connected: false,
            is_first: false,
            is_last: true,
            next_hop_stream: None,
            model_name: String::new(),
            model_url: String::new(),
            model_hash: String::new(),
            llama_server_started: false,
            rpc_server_started: false,
            setup_started: false,
        }
    }
}