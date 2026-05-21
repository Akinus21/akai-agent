use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::{auth, config::Config};

#[derive(Clone)]
pub struct QueueClient {
    base_url:   String,
    username:   String,
    client:     Client,
}

#[derive(Serialize)]
struct AuthRegisterRequest<'a> {
    username:    &'a str,
    worker_name: &'a str,
    public_key:  &'a str,
}

#[derive(Serialize)]
struct AuthDuoRequest<'a> {
    username:    &'a str,
    worker_name: &'a str,
    public_key:  &'a str,
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
    #[serde(default)]
    models:     Vec<String>,
}

#[derive(Serialize)]
struct HeartbeatRequest {
    gpu:      bool,
    vram_gb:  f64,
    rpc_port:  u16,
    alive:     bool,
    #[serde(default)]
    models:   Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct HeartbeatResponse {
    #[serde(default)]
    pub hub_commit: String,
}

#[derive(Deserialize, Debug)]
pub struct ProvisionResponse {
    #[serde(alias = "wg_private_key")]
    pub private_key:         Option<String>,
    pub wg_ip:               Option<String>,
    pub peer_id:             Option<String>,
    pub server_public_key:   Option<String>,
    #[serde(alias = "server_endpoint")]
    pub endpoint:            Option<String>,
    pub dns:                 Option<String>,
    pub allowed_ips:         Option<String>,
    #[serde(alias = "wg_preshared_key")]
    pub preshared_key:       Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct WorkerStatus {
    pub id:       String,
    pub alive:    bool,
    pub wg_ip:    String,
    pub rpc_port: u16,
}

impl QueueClient {
    pub fn new(base_url: &str, username: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            username: username.to_string(),
            client:   Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap(),
        }
    }

    pub fn from_config(cfg: &Config) -> Self {
        Self::new(&cfg.queue_url, &cfg.username)
    }

    fn timestamp() -> String {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        format!("{}", millis)
    }

    fn sign(&self, method: &str, path: &str, body: &[u8]) -> Result<(String, String)> {
        let private_key = auth::load_private_key()?;
        let ts = Self::timestamp();
        let sig = auth::sign_request(&private_key, &ts, method, path, body)?;
        Ok((ts, sig))
    }

    fn auth_headers(&self, method: &str, path: &str, body: &[u8]) -> Result<reqwest::header::HeaderMap> {
        let (ts, sig) = self.sign(method, path, body)?;
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("X-Akai-Username", self.username.parse().context("Invalid username header")?);
        headers.insert("X-Akai-Timestamp", ts.parse().context("Invalid timestamp header")?);
        headers.insert("X-Akai-Signature", sig.parse().context("Invalid signature header")?);
        Ok(headers)
    }

    pub async fn auth_register(&self, worker_name: &str, public_key: &str) -> Result<ProvisionResponse> {
        let body = serde_json::to_vec(&AuthRegisterRequest {
            username: &self.username,
            worker_name,
            public_key,
        })?;
        let path = "/auth/register";
        let (ts, sig) = self.sign("POST", path, &body)?;

        let resp = self.client
            .post(format!("{}{}", self.base_url, path))
            .header("X-Akai-Username", &self.username)
            .header("X-Akai-Timestamp", &ts)
            .header("X-Akai-Signature", &sig)
            .header("Content-Type", "application/json")
            .body(body)
            .send().await?;

        if resp.status() == 401 {
            let detail = resp.text().await.unwrap_or_default();
            bail!("AUTH_REQUIRED:{}", detail);
        }
        if !resp.status().is_success() {
            bail!("auth/register failed: {} — {}", resp.status(), resp.text().await?);
        }
        Ok(resp.json().await?)
    }

    pub async fn auth_duo(&self, worker_name: &str, public_key: &str) -> Result<ProvisionResponse> {
        let body = serde_json::to_vec(&AuthDuoRequest {
            username: &self.username,
            worker_name,
            public_key,
        })?;

        println!("  Duo push sent to {} — check your phone...", self.username);

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap();

        let resp = client
            .post(format!("{}/auth/duo", self.base_url))
            .header("Content-Type", "application/json")
            .body(body)
            .send().await?;

        if resp.status() == 401 {
            let detail = resp.text().await.unwrap_or_default();
            bail!("Duo 2FA denied: {}", detail);
        }
        if !resp.status().is_success() {
            bail!("auth/duo failed: {} — {}", resp.status(), resp.text().await?);
        }
        Ok(resp.json().await?)
    }

    pub async fn register(
        &self,
        id: &str, name: &str, wg_ip: &str, wg_peer_id: &str,
        gpu: bool, vram_gb: f64, rpc_port: u16,
    ) -> Result<()> {
        let body = serde_json::to_vec(&RegisterRequest { id, name, wg_ip, wg_peer_id, gpu, vram_gb, rpc_port, models: Vec::new() })?;
        let path = "/workers/register".to_string();
        let headers = self.auth_headers("POST", &path, &body)?;

        let resp = self.client
            .post(format!("{}{}", self.base_url, &path))
            .headers(headers)
            .header("Content-Type", "application/json")
            .body(body)
            .send().await?;
        if !resp.status().is_success() {
            bail!("register failed: {} — {}", resp.status(), resp.text().await?);
        }
        Ok(())
    }

    pub async fn heartbeat(
        &self, worker_id: &str, gpu: bool, vram_gb: f64, rpc_port: u16,
    ) -> Result<HeartbeatResponse> {
        let body = serde_json::to_vec(&HeartbeatRequest { gpu, vram_gb, rpc_port, alive: true, models: Vec::new() })?;
        let path = format!("/workers/{}/heartbeat", worker_id);
        let headers = self.auth_headers("POST", &path, &body)?;

        let resp = self.client
            .post(format!("{}{}", self.base_url, &path))
            .headers(headers)
            .header("Content-Type", "application/json")
            .body(body)
            .send().await?;
        if resp.status() == 404 {
            bail!("404: worker not found in registry");
        }
        if !resp.status().is_success() {
            bail!("heartbeat failed: {} — {}", resp.status(), resp.text().await?);
        }
        Ok(resp.json().await?)
    }

    pub async fn deregister(&self, worker_id: &str) -> Result<()> {
        let path = format!("/workers/{}", worker_id);
        let headers = self.auth_headers("DELETE", &path, &[])?;

        let _ = self.client
            .delete(format!("{}{}", self.base_url, &path))
            .headers(headers)
            .send().await?;
        Ok(())
    }

    pub async fn get_worker(&self, worker_id: &str) -> Result<WorkerStatus> {
        let path = format!("/workers/{}", worker_id);
        let headers = self.auth_headers("GET", &path, &[])?;

        let resp = self.client
            .get(format!("{}{}", self.base_url, &path))
            .headers(headers)
            .send().await?;
        if !resp.status().is_success() {
            bail!("status failed: {}", resp.status());
        }
        Ok(resp.json().await?)
    }
}