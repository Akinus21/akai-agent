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
    #[command(about = "Install akai-agent as a systemd service")]
    Install,

    #[command(about = "Uninstall the systemd service")]
    Uninstall,

    #[command(about = "Register and start the worker (clean first if flags provided)")]
    Init {
        #[arg(long)]
        hub: Option<String>,

        #[arg(long)]
        username: Option<String>,

        #[arg(long)]
        api_url: Option<String>,
    },

    #[command(about = "Remove all local data and VPN config")]
    Clean,

    #[command(about = "Stop the running worker")]
    Stop,

    #[command(about = "Upgrade akai-agent via Homebrew")]
    Upgrade,
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Install => crate::service::install_service(),
        Commands::Uninstall => crate::service::uninstall_service(),
        Commands::Init { hub, username, api_url } => {
            let has_flags = hub.is_some() || username.is_some() || api_url.is_some();
            if has_flags {
                handlers::clean()?;
                handlers::init_and_start(
                    hub.as_deref().unwrap_or_default(),
                    username.as_deref().unwrap_or_default(),
                    api_url.as_deref(),
                ).await
            } else {
                handlers::resume_and_start().await
            }
        }
        Commands::Clean => handlers::clean(),
        Commands::Stop  => handlers::stop(),
        Commands::Upgrade => handlers::upgrade(),
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

        let _ = std::process::Command::new("wg-quick")
            .args(["down", "wg0"])
            .output();

        let wireguard_conf = "/etc/wireguard/wg0.conf";
        if std::path::Path::new(wireguard_conf).exists() {
            std::fs::remove_file(wireguard_conf)?;
            println!("  Removed {}", wireguard_conf);
        }

        println!("Clean complete.");
        Ok(())
    }

    pub async fn init_and_start(hub: &str, username: &str, api_url: Option<&str>) -> Result<()> {
        if hub.is_empty() || username.is_empty() {
            anyhow::bail!("--hub and --username are required for initial setup");
        }

        let device_name = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "akai-worker".to_string());
        let worker_id = format!("{}:{}", username, device_name);

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

        start_worker(&cfg).await
    }

    pub async fn resume_and_start() -> Result<()> {
        let cfg = config::load_config()
            .context("No saved config found. Run: akai-agent init --hub <addr> --username <user>")?;

        if cfg.hub_addr.is_empty() || cfg.username.is_empty() {
            anyhow::bail!("Config incomplete. Run: akai-agent init --hub <addr> --username <user>");
        }

        println!("Resuming akai-agent worker");
        println!("  Hub:      {}", cfg.hub_addr);
        if !cfg.hub_api_url.is_empty() {
            println!("  API URL: {}", cfg.hub_api_url);
        }
        println!("  Worker:   {}", cfg.worker_id);

        start_worker(&cfg).await
    }

    async fn start_worker(cfg: &config::Config) -> Result<()> {
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
            let mut vpn_addr = None;
            for attempt in 1..=20 {
                match enroll_vpn(&cfg.hub_api_url, &cfg.worker_id, &cfg.username).await {
                    Ok(addr) => {
                        println!("  VPN:    enrolled, hub at {}", addr);
                        vpn_addr = Some(addr);
                        break;
                    }
                    Err(e) => {
                        eprintln!("VPN enrollment attempt {}/20 failed: {}", attempt, e);
                        if attempt < 20 {
                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        }
                    }
                }
            }
            match vpn_addr {
                Some(addr) => addr,
                None => {
                    eprintln!("All 20 VPN enrollment attempts failed, falling back to direct connection to {}", hub_addr);
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

    pub fn upgrade() -> Result<()> {
        println!("Upgrading akai-agent...");

        println!("Running brew update...");
        let update = std::process::Command::new("brew")
            .arg("update")
            .status()
            .context("brew not found — is Homebrew installed?")?;
        if !update.success() {
            anyhow::bail!("brew update failed");
        }

        println!("Running brew upgrade akai-agent...");
        let upgrade = std::process::Command::new("brew")
            .arg("upgrade")
            .arg("akai-agent")
            .status()?;
        if !upgrade.success() {
            anyhow::bail!("brew upgrade akai-agent failed");
        }

        println!("Upgrade complete.");
        Ok(())
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

        // Bring down existing WireGuard if up
        let _ = std::process::Command::new("wg-quick")
            .args(["down", "wg0"])
            .output();

        // Write WireGuard config (split tunnel — only VPN traffic)
        // Model downloads go through hub's /model/download proxy over VPN
        let config_text = config_text
            .replace("AllowedIPs = 0.0.0.0/0, ::/0", "AllowedIPs = 10.8.0.0/24")
            .replace("AllowedIPs = 0.0.0.0/0", "AllowedIPs = 10.8.0.0/24")
            .replace("PersistentKeepalive = 0", "PersistentKeepalive = 25")
            .lines()
            .filter(|line| !line.trim().starts_with("DNS ="))
            .collect::<Vec<_>>()
            .join("\n");
        let wg_conf = std::path::Path::new("/etc/wireguard/wg0.conf");
        std::fs::write(wg_conf, &config_text)?;
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