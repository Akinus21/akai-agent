use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub queue_url:   String,
    pub api_key:     String,
    pub worker_id:   String,
    pub worker_name: String,
    pub wg_ip:       String,
    pub wg_peer_id:  String,
    pub rpc_port:    u16,
    pub gpu:         bool,
    pub vram_gb:     f64,
    pub rpc_binary:  String,
    pub rpc_version: String,
    #[serde(default)]
    pub gpu_backend: String,
    #[serde(default)]
    pub username:    String,
    #[serde(default)]
    pub public_key:  String,
    #[serde(default)]
    pub tunnel_host: String,
    #[serde(default)]
    pub tunnel_port: u16,
    #[serde(default)]
    pub hub_wg_ip: String,
    #[serde(default)]
    pub hub_port: u16,
    #[serde(default)]
    pub petals_model: String,
    #[serde(default)]
    pub hub_addr: String,
}

impl Config {
    pub fn get_hub_addr(&self) -> String {
        std::env::var("HUB_ADDR")
            .unwrap_or_else(|_| {
                if !self.hub_addr.is_empty() {
                    self.hub_addr.clone()
                } else if !self.hub_wg_ip.is_empty() {
                    format!("{}:{}", self.hub_wg_ip, self.hub_port)
                } else {
                    "127.0.0.1:50051".to_string()
                }
            })
    }
}

pub fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("akai-agent")
        .join("config.toml")
}

pub fn data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("akai-agent")
}

pub fn key_dir() -> PathBuf {
    data_dir()
}

pub fn private_key_path() -> PathBuf {
    key_dir().join("id_akai")
}

pub fn public_key_path() -> PathBuf {
    key_dir().join("id_akai.pub")
}

pub fn save_config(config: &Config) -> Result<()> {
    let path = config_path();
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, toml::to_string_pretty(config)?)?;
    Ok(())
}

pub fn load_config() -> Result<Config> {
    let content = std::fs::read_to_string(config_path())
        .with_context(|| format!(
            "Config not found at {}. Run `akai-agent init` first.",
            config_path().display()
        ))?;
    Ok(toml::from_str(&content)?)
}