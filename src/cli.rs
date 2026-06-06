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
    },

    Clean,

    Start,

    Stop,
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { hub, username } =>
            handlers::init(&hub, &username).await,
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

    pub async fn init(hub: &str, username: &str) -> Result<()> {
        let device_name = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "akai-worker".to_string());
        let worker_id = format!("{}:{}", username, device_name);

        println!("Initializing akai-agent worker");
        println!("  Hub:      {}", hub);
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
            tunnel_ca_cert: String::new(),
            tunnel_client_cert: String::new(),
            tunnel_client_key: String::new(),
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

        // Fetch tunnel certs from hub HTTP API (akai-hub.akinus21.com over HTTPS)
        // Then connect to tunnel.akinus21.com:443 with mTLS for data
        let (ca_cert, client_cert, client_key, tunnel_addr) = if !cfg.tunnel_ca_cert.is_empty() {
            // Already have certs, derive tunnel addr from hub addr
            let t_addr = derive_tunnel_addr(&hub_addr);
            (cfg.tunnel_ca_cert.as_bytes().to_vec(), cfg.tunnel_client_cert.as_bytes().to_vec(), cfg.tunnel_client_key.as_bytes().to_vec(), t_addr)
        } else {
            match fetch_tunnel_certs_from_hub(&hub_addr).await {
                Ok((ca, cert, key, t_addr)) => {
                    println!("  Tunnel: certs fetched, connecting via {}", t_addr);
                    (ca, cert, key, t_addr)
                }
                Err(e) => {
                    println!("  Tunnel: no certs available ({}), using raw TCP", e);
                    (Vec::new(), Vec::new(), Vec::new(), hub_addr.clone())
                }
            }
        };

        // If tunnel certs available, connect through tunnel; otherwise raw TCP to hub
        let data_addr = if !ca_cert.is_empty() { tunnel_addr } else { hub_addr.clone() };

        let pid_file = std::path::Path::new("/run/akai-agent.pid");
        std::fs::write(pid_file, std::process::id().to_string())?;
        println!("PID: {} (saved to {})", std::process::id(), pid_file.display());

        worker::run_hub_worker(worker::HubWorkerConfig {
            hub_addr: data_addr,
            worker_id: cfg.worker_id.clone(),
            has_gpu: gpu_info.has_gpu,
            vram_gb: gpu_info.vram_gb as f32,
            rpc_port: cfg.rpc_port,
            llama_port: cfg.llama_port,
            tunnel_ca_cert: ca_cert,
            tunnel_client_cert: client_cert,
            tunnel_client_key: client_key,
        }).await
    }

    fn derive_tunnel_addr(hub_addr: &str) -> String {
        // Replace "akai-hub" with "tunnel" in the hostname
        let parts: Vec<&str> = hub_addr.splitn(2, ':').collect();
        let host = parts[0].replace("akai-hub", "tunnel");
        if parts.len() > 1 {
            format!("{}:{}", host, parts[1])
        } else {
            host
        }
    }

async fn fetch_tunnel_certs_from_hub(hub_addr: &str) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, String)> {
    let host = hub_addr.split(':').next().unwrap_or("127.0.0.1");
    let url = format!("https://{}/tunnel/certs", host);
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("hub returned {}", resp.status());
    }
    let json: serde_json::Value = resp.json().await?;
    let ca = json["ca_cert"].as_str().unwrap_or("").as_bytes().to_vec();
    let cert = json["worker_cert"].as_str().unwrap_or("").as_bytes().to_vec();
    let key = json["worker_key"].as_str().unwrap_or("").as_bytes().to_vec();
    if ca.is_empty() {
        anyhow::bail!("no ca_cert in response");
    }
    let tunnel_host = json["tunnel_host"].as_str().unwrap_or("tunnel.akinus21.com");
    let tunnel_port = json["tunnel_port"].as_u64().unwrap_or(443);
    let tunnel_addr = format!("{}:{}", tunnel_host, tunnel_port);
    Ok((ca, cert, key, tunnel_addr))
}
}