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
        queue: String,

        #[arg(long)]
        key: String,

        #[arg(long)]
        name: Option<String>,

        #[arg(long, default_value = "50052")]
        rpc_port: u16,
    },

    Start,

    Install,

    Status,

    UpdateRpc,
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { queue, key, name, rpc_port } =>
            handlers::init(&queue, &key, name, rpc_port).await,
        Commands::Start      => handlers::start().await,
        Commands::Install    => handlers::install().await,
        Commands::Status     => handlers::status().await,
        Commands::UpdateRpc  => handlers::update_rpc().await,
    }
}

mod handlers {
    use anyhow::{Context, Result};
    use std::time::Duration;
    use crate::{config, gpu, queue_client::QueueClient, rpc, wireguard};

    pub async fn init(
        queue_url: &str,
        api_key:   &str,
        name:      Option<String>,
        rpc_port:  u16,
    ) -> Result<()> {
        let worker_name = name.unwrap_or_else(||
            hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "akai-worker".to_string())
        );

        println!("Initializing akai-agent as \"{}\"", worker_name);

        let gpu_info = gpu::detect_gpu();
        if gpu_info.has_gpu {
            println!("GPU: {} ({:.1} GB VRAM)", gpu_info.name, gpu_info.vram_gb);
        } else {
            println!("No GPU detected (CPU-only worker)");
        }

        let client = QueueClient::new(queue_url, api_key);
        println!("Requesting WireGuard peer from {}...", queue_url);
        let provision = client.provision(&worker_name).await
            .context("Failed to provision WireGuard peer")?;
        println!("Assigned WireGuard IP: {}", provision.wg_ip);

        println!("Configuring WireGuard...");
        wireguard::configure(&provision).await
            .context("WireGuard configuration failed")?;
        println!("WireGuard tunnel up ({})", provision.wg_ip);

        println!("Downloading rpc-server...");
        let rpc_path = rpc::ensure_rpc_server().await
            .context("Failed to download rpc-server binary")?;
        let rpc_version = rpc::current_version();
        println!("rpc-server ready at {}", rpc_path.display());

        let worker_id = worker_name.clone();
        match client.register(
            &worker_id,
            &worker_name,
            &provision.wg_ip,
            &provision.peer_id,
            gpu_info.has_gpu,
            gpu_info.vram_gb,
            rpc_port,
        ).await {
            Ok(_)  => println!("Registered with queue"),
            Err(e) => eprintln!("Registration failed: {}. Retry with `akai-agent start`.", e),
        }

        let cfg = config::Config {
            queue_url:   queue_url.to_string(),
            api_key:     api_key.to_string(),
            worker_id:   worker_id.clone(),
            worker_name: worker_name.clone(),
            wg_ip:       provision.wg_ip.clone(),
            wg_peer_id:  provision.peer_id.clone(),
            rpc_port,
            gpu:         gpu_info.has_gpu,
            vram_gb:     gpu_info.vram_gb,
            rpc_binary:  rpc_path.to_string_lossy().to_string(),
            rpc_version,
        };
        config::save_config(&cfg)?;
        println!("Config saved to {}", config::config_path().display());

        println!();
        println!("Initialization complete!");
        println!("   Run:  akai-agent start");
        println!("   Or:   sudo akai-agent install");
        Ok(())
    }

    pub async fn start() -> Result<()> {
        let cfg = config::load_config()
            .context("Config not found. Run `akai-agent init` first.")?;

        println!("Starting akai-agent");
        println!("  Worker:    {}", cfg.worker_id);
        println!("  Queue:     {}", cfg.queue_url);
        println!("  WireGuard: {}", cfg.wg_ip);
        println!("  RPC port:  {}", cfg.rpc_port);

        let client = QueueClient::new(&cfg.queue_url, &cfg.api_key);

        {
            let client  = client.clone();
            let id      = cfg.worker_id.clone();
            tokio::spawn(async move {
                tokio::signal::ctrl_c().await.ok();
                eprintln!("Shutting down — deregistering...");
                let _ = client.deregister(&id).await;
                std::process::exit(0);
            });
        }

        let rpc_path = rpc::ensure_rpc_server().await
            .context("rpc-server binary not found")?;

        let mut child = rpc::spawn_rpc_server(&rpc_path, cfg.rpc_port)
            .context("Failed to spawn rpc-server")?;
        println!("rpc-server running on 0.0.0.0:{}", cfg.rpc_port);

        let mut heartbeat_tick  = tokio::time::interval(Duration::from_secs(30));
        let mut update_tick     = tokio::time::interval(Duration::from_secs(86400));
        heartbeat_tick.tick().await;
        update_tick.tick().await;

        loop {
            tokio::select! {
                _ = heartbeat_tick.tick() => {
                    if matches!(child.try_wait(), Ok(Some(_))) {
                        eprintln!("rpc-server exited — respawning...");
                        child = rpc::spawn_rpc_server(&rpc_path, cfg.rpc_port)?;
                        println!("rpc-server restarted on :{}", cfg.rpc_port);
                    }

                    match client.heartbeat(
                        &cfg.worker_id, cfg.gpu, cfg.vram_gb, cfg.rpc_port
                    ).await {
                        Ok(_) => tracing::info!("Heartbeat OK"),
                        Err(e) if is_not_found(&e) => {
                            eprintln!("Not in registry — re-registering...");
                            let _ = client.register(
                                &cfg.worker_id, &cfg.worker_name,
                                &cfg.wg_ip,     &cfg.wg_peer_id,
                                cfg.gpu,         cfg.vram_gb, cfg.rpc_port,
                            ).await;
                        }
                        Err(e) => eprintln!("Heartbeat failed: {e}"),
                    }
                }

                _ = update_tick.tick() => {
                    match rpc::needs_update(&cfg.rpc_version).await {
                        Ok(true) => {
                            println!("New rpc-server available — updating...");
                            child.kill().ok();
                            child.wait().ok();
                            if let Err(e) = rpc::download_latest().await {
                                eprintln!("Update failed: {e}");
                            } else {
                                child = rpc::spawn_rpc_server(&rpc_path, cfg.rpc_port)?;
                                println!("rpc-server updated and restarted");
                            }
                        }
                        Ok(false) => tracing::debug!("rpc-server up to date"),
                        Err(e)    => tracing::warn!("Update check failed: {e}"),
                    }
                }
            }
        }
    }

    pub async fn install() -> Result<()> {
        crate::service::install()
    }

    pub async fn status() -> Result<()> {
        let cfg = config::load_config()
            .context("Config not found. Run `akai-agent init` first.")?;
        let client = QueueClient::new(&cfg.queue_url, &cfg.api_key);
        match client.get_worker(&cfg.worker_id).await {
            Ok(info) => {
                println!("Worker:    {}", cfg.worker_id);
                println!("Queue:     {}", cfg.queue_url);
                println!("WireGuard: {}", cfg.wg_ip);
                println!("RPC port:  {}", cfg.rpc_port);
                println!("Status:    {}", if info.alive { "alive" } else { "dead" });
            }
            Err(e) => eprintln!("Could not reach queue: {e}"),
        }
        Ok(())
    }

    pub async fn update_rpc() -> Result<()> {
        println!("Checking for rpc-server updates...");
        let cfg = config::load_config().ok();
        let current = cfg.as_ref().map(|c| c.rpc_version.as_str()).unwrap_or("unknown");

        match rpc::needs_update(current).await? {
            true => {
                println!("Downloading latest rpc-server...");
                rpc::download_latest().await?;
                println!("rpc-server updated. Restart `akai-agent start` to use it.");
            }
            false => println!("rpc-server is already up to date ({})", current),
        }
        Ok(())
    }

    fn is_not_found(e: &anyhow::Error) -> bool {
        e.to_string().contains("404")
    }
}