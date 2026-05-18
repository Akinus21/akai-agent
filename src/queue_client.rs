use anyhow::{bail, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct QueueClient {
    base_url: String,
    api_key:  String,
    client:   Client,
}

#[derive(Serialize)]
struct ProvisionRequest<'a> {
    name: &'a str,
}

#[derive(Serialize)]
struct RegisterRequest<'a> {
    id:         &'a str,
    name:       &'a str,
    wg_ip:      &'a str,
    wg_peer_id: &'a str,
    gpu:        bool,
    vram_gb:    f64,
    rpc_port:   u16,
}

#[derive(Serialize)]
struct HeartbeatRequest {
    gpu:      bool,
    vram_gb:  f64,
    rpc_port: u16,
    alive:    bool,
}

#[derive(Deserialize, Debug)]
pub struct ProvisionResponse {
    #[serde(alias = "wg_private_key")]
    pub private_key:         Option<String>,
    pub wg_ip:               Option<String>,
    pub peer_id:             Option<String>,
    pub server_public_key:   Option<String>,
    pub endpoint:             Option<String>,
    pub dns:                 Option<String>,
    pub allowed_ips:         Option<String>,
    #[serde(alias = "wg_server_endpoint")]
    pub wg_server_endpoint:   Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct WorkerStatus {
    pub id:       String,
    pub alive:    bool,
    pub wg_ip:    String,
    pub rpc_port: u16,
}

impl QueueClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key:  api_key.to_string(),
            client:   Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
        }
    }

    fn auth(&self) -> (&'static str, String) {
        ("X-Worker-Key", self.api_key.clone())
    }

    pub async fn provision(&self, name: &str) -> Result<ProvisionResponse> {
        let resp = self.client
            .post(format!("{}/workers/provision", self.base_url))
            .header(self.auth().0, self.auth().1)
            .json(&ProvisionRequest { name })
            .send().await?;
        if !resp.status().is_success() {
            bail!("provision failed: {} — {}", resp.status(), resp.text().await?);
        }
        Ok(resp.json().await?)
    }

    pub async fn register(
        &self,
        id: &str, name: &str, wg_ip: &str, wg_peer_id: &str,
        gpu: bool, vram_gb: f64, rpc_port: u16,
    ) -> Result<()> {
        let resp = self.client
            .post(format!("{}/workers/register", self.base_url))
            .header(self.auth().0, self.auth().1)
            .json(&RegisterRequest { id, name, wg_ip, wg_peer_id, gpu, vram_gb, rpc_port })
            .send().await?;
        if !resp.status().is_success() {
            bail!("register failed: {} — {}", resp.status(), resp.text().await?);
        }
        Ok(())
    }

    pub async fn heartbeat(
        &self, worker_id: &str, gpu: bool, vram_gb: f64, rpc_port: u16,
    ) -> Result<()> {
        let resp = self.client
            .post(format!("{}/workers/{}/heartbeat", self.base_url, worker_id))
            .header(self.auth().0, self.auth().1)
            .json(&HeartbeatRequest { gpu, vram_gb, rpc_port, alive: true })
            .send().await?;
        if resp.status() == 404 {
            bail!("404: worker not found in registry");
        }
        if !resp.status().is_success() {
            bail!("heartbeat failed: {} — {}", resp.status(), resp.text().await?);
        }
        Ok(())
    }

    pub async fn deregister(&self, worker_id: &str) -> Result<()> {
        let _ = self.client
            .delete(format!("{}/workers/{}", self.base_url, worker_id))
            .header(self.auth().0, self.auth().1)
            .send().await?;
        Ok(())
    }

    pub async fn get_worker(&self, worker_id: &str) -> Result<WorkerStatus> {
        let resp = self.client
            .get(format!("{}/workers/{}", self.base_url, worker_id))
            .header(self.auth().0, self.auth().1)
            .send().await?;
        if !resp.status().is_success() {
            bail!("status failed: {}", resp.status());
        }
        Ok(resp.json().await?)
    }
}