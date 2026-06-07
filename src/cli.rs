use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "akai-agent")]
#[command(about = "Remote GPU worker for the akai-net distributed inference system")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Init {
        #[arg(long)]
        hub: String,

        #[arg(long)]
        username: String,

        #[arg(long)]
        api_url: Option<String>,
    },

    Clean,

    Start,

    Stop,
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { hub, username, api_url } =>
            handlers::init(&hub, &username, api_url.as_deref()).await,
        Commands::Clean       => handlers::clean(),
        Commands::Start       => handlers::start().await,
        Commands::Stop        => handlers::stop(),
    }
}

mod handlers {
    use anyhow::{Context, Result};
    use crate::{config, gpu, worker};

    pub fn clean() -> Result<()> {
        println!("Cleaning up akai-agent data...");

        let config_dirs = [
            "/root/.config/akai-agent",
            "/root/.local/share/akai-agent",
        ];
        for dir in &config_dirs {
            if std::path::Path::new(dir).exists() {
                std::fs::remove_dir_all(dir)?;
                println!("  Removed {}", dir);
            }
        }

        let wireguard_conf = "/etc/wireguard/wg0.conf";
        if std::path::Path::new(wireguard_conf).exists() {
            std::fs::remove_file(wireguard_conf)?;
            println!("  Removed {}", wireguard_conf);
        }

        println!("Clean complete.");
        Ok(())
    }

    pub async fn init(hub: &str, username: &str, api_url: Option<&str>) -> Result<()> {
        let device_name = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "akai-worker".to_string());
        let worker_id = format!("{}:{}", username, device_name);

        // hub is the VPN address (e.g. 10.8.0.1:50051)
        // api_url is the public HTTP API for enrollment (e.g. http://akai-hub.akinus21.com:8080)
        let hub_api = api_url.unwrap_or("").to_string();

        println!("Initializing akai-agent worker");
        println!("  Hub:      {}", hub);
        if !hub_api.is_empty() {
            println!("  API URL: {}", hub_api);
        }
        println!("  Worker:   {}", worker_id);

        let gpu_info = gpu::detect_gpu();
        if gpu_info.has_gpu {
            println!("  GPU:      {} ({:.1} GB)", gpu_info.name, gpu_info.vram_gb);
        } else {
            println!("  GPU:      CPU only");
        }

        let cfg = config::Config {
            queue_url:   String::new(),
            api_key:     String::new(),
            worker_id:   worker_id.clone(),
            worker_name: device_name.clone(),
            wg_ip:       String::new(),
            wg_peer_id:  String::new(),
            rpc_port:    50052,
            llama_port:  8081,
            gpu:         gpu_info.has_gpu,
            vram_gb:     gpu_info.vram_gb,
            gpu_backend: gpu_info.backend.to_string(),
            rpc_binary:  String::new(),
            rpc_version: String::new(),
            username:    username.to_string(),
            public_key:  String::new(),
            tunnel_host: String::new(),
            tunnel_port: 0,
            hub_wg_ip:   String::new(),
            hub_port:    50051,
            petals_model: String::new(),
            hub_addr:    hub.to_string(),
            hub_api_url: hub_api,
        };

        config::save_config(&cfg)?;
        println!("Config saved to {}", config::config_path().display());

        println!();
        println!("Initialization complete!");
        println!("   Run:  sudo akai-agent start");
        Ok(())
    }

    pub fn stop() -> Result<()> {
        let pid_file = std::path::Path::new("/run/akai-agent.pid");
        if pid_file.exists() {
            let pid: i32 = std::fs::read_to_string(pid_file)?
                .trim()
                .parse()
                .context("Invalid PID file")?;
            println!("Stopping akai-agent (PID {})...", pid);
            std::process::Command::new("kill")
                .arg(pid.to_string())
                .status()?;
            std::fs::remove_file(pid_file)?;
            println!("Stopped.");
        } else {
            println!("Not running (no PID file).");
        }
        Ok(())
    }

    pub async fn start() -> Result<()> {
        let cfg = config::load_config()
            .context("Config not found. Run `akai-agent init` first.")?;

        let hub_addr = if !cfg.hub_addr.is_empty() {
            cfg.hub_addr.clone()
        } else {
            std::env::var("HUB_ADDR").context("HUB_ADDR not set and no hub_addr in config")?
        };

        println!("Starting akai-agent worker");
        println!("  Hub:    {}", hub_addr);
        println!("  Worker: {}", cfg.worker_id);

        let gpu_info = gpu::detect_gpu();
        if gpu_info.has_gpu {
            println!("  GPU:    {} ({:.1} GB)", gpu_info.name, gpu_info.vram_gb);
        } else {
            println!("  GPU:    CPU only");
        }

        // Ensure WireGuard VPN is connected
        let vpn_connected = std::path::Path::new("/dev/net/tun").exists()
            && std::process::Command::new("wg")
                .arg("show")
                .arg("wg0")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);

        let hub_connect_addr = if vpn_connected {
            println!("  VPN:    already connected (wg0 up)");
            hub_addr.clone()
        } else if cfg.hub_api_url.is_empty() {
            eprintln!("VPN not connected and no hub API URL configured.");
            eprintln!("Run: akai-agent init --hub <vpn_addr> --username <user> --api-url <public_api_url>");
            anyhow::bail!("Cannot enroll VPN without hub API URL");
        } else {
            println!("  VPN:    not connected, enrolling via {}...", cfg.hub_api_url);
            match enroll_vpn(&cfg.hub_api_url, &cfg.worker_id, &cfg.username).await {
                Ok(vpn_addr) => {
                    println!("  VPN:    enrolled, hub at {}", vpn_addr);
                    vpn_addr
                }
                Err(e) => {
                    eprintln!("VPN enrollment failed: {}", e);
                    eprintln!("Falling back to direct connection to {}", hub_addr);
                    hub_addr.clone()
                }
            }
        };

        let pid_file = std::path::Path::new("/run/akai-agent.pid");
        std::fs::write(pid_file, std::process::id().to_string())?;
        println!("PID: {} (saved to {})", std::process::id(), pid_file.display());

        worker::run_hub_worker(worker::HubWorkerConfig {
            hub_addr: hub_connect_addr,
            worker_id: cfg.worker_id.clone(),
            has_gpu: gpu_info.has_gpu,
            vram_gb: gpu_info.vram_gb as f32,
            rpc_port: cfg.rpc_port,
            llama_port: cfg.llama_port,
        }).await
    }

    async fn enroll_vpn(hub_api_url: &str, worker_id: &str, username: &str) -> Result<String> {
        let url = format!("{}/auth/vpn", hub_api_url.trim_end_matches('/'));

        let client = reqwest::Client::new();
        let resp = client.post(&url)
            .json(&serde_json::json!({
                "username": username,
                "worker_name": worker_id.split(':').last().unwrap_or("worker"),
            }))
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .context("Failed to reach hub for VPN enrollment")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Hub returned {}: {}", status, body);
        }

        let json: serde_json::Value = resp.json().await?;
        let config_text = json["wireguard_config"].as_str()
            .ok_or_else(|| anyhow::anyhow!("missing wireguard_config in response"))?;
        let hub_vpn_addr = json["hub_vpn_addr"].as_str()
            .ok_or_else(|| anyhow::anyhow!("missing hub_vpn_addr in response"))?;

        // Write WireGuard config (modify for full tunnel so DNS works)
        let config_text = config_text
            .replace("AllowedIPs = 10.8.0.0/24", "AllowedIPs = 0.0.0.0/0, ::/0")
            .replace("PersistentKeepalive = 0", "PersistentKeepalive = 25");
        std::fs::write(&wg_conf, &config_text)?;
        println!("  VPN:    config written to {}", wg_conf.display());

        // Bring up WireGuard
        let output = std::process::Command::new("wg-quick")
            .arg("up")
            .arg("wg0")
            .output()
            .context("wg-quick not found — is wireguard-tools installed?")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("wg-quick up wg0 failed: {}", stderr);
        }
        println!("  VPN:    WireGuard interface wg0 brought up");

        Ok(hub_vpn_addr.to_string())
    }
}