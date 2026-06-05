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
    use std::fs;
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

        let mut cfg = config::Config {
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

        let pid_file = std::path::Path::new("/run/akai-agent.pid");
        std::fs::write(pid_file, std::process::id().to_string())?;
        println!("PID: {} (saved to {})", std::process::id(), pid_file.display());

        worker::run_hub_worker(worker::HubWorkerConfig {
            hub_addr,
            worker_id: cfg.worker_id.clone(),
            has_gpu: gpu_info.has_gpu,
            vram_gb: gpu_info.vram_gb as f32,
            rpc_port: cfg.rpc_port,
            llama_port: cfg.llama_port,
        }).await
    }
}