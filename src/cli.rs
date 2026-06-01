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
        username: String,

        #[arg(long)]
        name: Option<String>,

        #[arg(long, default_value = "50052")]
        rpc_port: u16,

        #[arg(long)]
        hub_wg_ip: Option<String>,

        #[arg(long)]
        hub_port: Option<u16>,
    },

    Clean,

    Start,

    StartWorker {
        #[arg(long)]
        hub_addr: String,

        #[arg(long)]
        worker_id: String,

        #[arg(long)]
        model_path: String,

        #[arg(long, default_value = "0")]
        layer_offset: usize,

        #[arg(long, default_value = "32")]
        num_layers: usize,
    },

    Install,

    Status,

    UpdateRpc,

    PetalsStart {
        #[arg(long)]
        model: String,

        #[arg(long, default_value = "50052")]
        port: u16,

        #[arg(long)]
        quantize: Option<String>,
    }
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { queue, username, name, rpc_port, hub_wg_ip, hub_port } =>
            handlers::init(&queue, &username, name, rpc_port, hub_wg_ip, hub_port).await,
        Commands::Clean       => handlers::clean(),
        Commands::Start       => handlers::start().await,
        Commands::StartWorker { hub_addr, worker_id, model_path, layer_offset, num_layers } =>
            handlers::start_worker(&hub_addr, &worker_id, &model_path, layer_offset, num_layers).await,
        Commands::Install     => handlers::install().await,
        Commands::Status      => handlers::status().await,
        Commands::UpdateRpc   => handlers::update_rpc().await,
        Commands::PetalsStart { model, port, quantize } =>
            handlers::start_petals(&model, port, quantize).await,
    }
}

mod handlers {
    use anyhow::{Context, Result};
    use std::process::Stdio;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use crate::{auth, config, gpu, petals, queue_client::QueueClient, rpc, wireguard, worker};

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

    pub async fn init(
        queue_url: &str,
        username:  &str,
        name:      Option<String>,
        rpc_port:  u16,
        hub_wg_ip: Option<String>,
        hub_port: Option<u16>,
    ) -> Result<()> {
        let device_name = name.unwrap_or_else(||
            hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "akai-worker".to_string())
        );
        let worker_id = format!("{}:{}", username, device_name);

        println!("Initializing akai-agent for user \"{}\" as \"{}\"", username, worker_id);

        let (_, public_key) = auth::ensure_keypair()?;

        let gpu_info = gpu::detect_gpu();
        if gpu_info.has_gpu {
            println!("GPU: {} ({:.1} GB VRAM, {})", gpu_info.name, gpu_info.vram_gb, gpu_info.backend);
        } else {
            println!("No GPU detected (CPU-only worker)");
        }

        let client = QueueClient::new(queue_url, username);
        println!("Authenticating as '{}' with queue at {}...", username, queue_url);

        let provision = match client.auth_register(&device_name, &public_key).await {
            Ok(p) => {
                println!("Authenticated with existing key.");
                p
            }
            Err(e) if e.to_string().starts_with("AUTH_REQUIRED:") => {
                println!("New worker — Duo 2FA required.");
                client.auth_duo(&device_name, &public_key).await
                    .context("Duo 2FA failed")?
            }
            Err(e) => return Err(e),
        };

        println!("Downloading rpc-server...");
        let rpc_path = rpc::ensure_rpc_server().await
            .context("Failed to download rpc-server binary")?;
        let rpc_version = rpc::current_version();
        println!("rpc-server ready at {}", rpc_path.display());

        let wg_ip = provision.wg_ip.clone().unwrap_or_default();
        let peer_id = provision.peer_id.clone().unwrap_or_default();

        match client.register(
            &worker_id,
            &device_name,
            &wg_ip,
            &peer_id,
            gpu_info.has_gpu,
            gpu_info.vram_gb,
            rpc_port,
            None,
        ).await {
            Ok(_)  => println!("Registered with queue"),
            Err(e) => eprintln!("Registration failed: {}. Retry with `akai-agent start`.", e),
        }

        let mut cfg = config::Config {
            queue_url:   queue_url.to_string(),
            api_key:     String::new(),
            worker_id:   worker_id.clone(),
            worker_name: device_name.clone(),
            wg_ip:       wg_ip.clone(),
            wg_peer_id:  peer_id.clone(),
            rpc_port,
            gpu:         gpu_info.has_gpu,
            vram_gb:     gpu_info.vram_gb,
            gpu_backend: gpu_info.backend.to_string(),
            rpc_binary:  rpc_path.to_string_lossy().to_string(),
            rpc_version,
            username:    username.to_string(),
            public_key,
            tunnel_host: String::new(),
            tunnel_port: 0,
            hub_wg_ip: hub_wg_ip.unwrap_or_default(),
            hub_port: hub_port.unwrap_or(8080),
            petals_model: String::new(),
            hub_addr: String::new(),
        };

        println!();
        println!("Fetching tunnel certificates...");
        match fetch_tunnel_certs(&mut cfg, &queue_url, &username).await {
            Ok(()) => {},
            Err(e) => eprintln!("Tunnel cert fetch failed: {}. Run `akai-agent start` to retry.", e),
        }

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

        // Check if HUB_ADDR is set for new architecture
        if let Ok(hub_addr) = std::env::var("HUB_ADDR") {
            println!("Starting akai-agent in hub mode");
            println!("  Hub:       {}", hub_addr);
            println!("  Worker:    {}", cfg.worker_id);

            let gpu_info = gpu::detect_gpu();
            if gpu_info.has_gpu {
                println!("  GPU:       {} ({:.1} GB)", gpu_info.name, gpu_info.vram_gb);
            } else {
                println!("  GPU:       CPU only");
            }

            worker::run_hub_worker(worker::HubWorkerConfig {
                hub_addr,
                worker_id: cfg.worker_id.clone(),
                has_gpu: gpu_info.has_gpu,
                vram_gb: gpu_info.vram_gb as f32,
                rpc_port: cfg.rpc_port,
            }).await
        } else {
            // Legacy queue-based start
            start_queue_worker(&cfg).await
        }
    }

    async fn start_queue_worker(cfg: &config::Config) -> Result<()> {
        println!("Starting akai-agent (queue mode)");
        println!("  Worker:    {}", cfg.worker_id);
        println!("  Queue:     {}", cfg.queue_url);
        println!("  RPC port:  {}", cfg.rpc_port);

        let use_tunnel = !cfg.tunnel_host.is_empty();
        let tunnel_connected: Arc<AtomicBool> = Arc::new(AtomicBool::new(!use_tunnel));

        if use_tunnel {
            println!("  Tunnel:    {}:{}", cfg.tunnel_host, cfg.tunnel_port);
        } else {
            println!("  WireGuard: {}", cfg.wg_ip);
        }

        if !use_tunnel {
            println!("Verifying WireGuard tunnel...");
            if !wireguard::check_tunnel(&cfg.wg_ip) {
                eprintln!("  WireGuard tunnel is down — re-establishing...");
                wireguard::ensure_tunnel(&cfg.wg_ip)
                    .context("Failed to establish WireGuard tunnel. RPC workers will be unreachable.")?;
            }
        }

        let client = QueueClient::from_config(cfg);

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

        if use_tunnel {
            let cert_dir = config::data_dir().join("tunnel-certs");
            let ca = std::fs::read(cert_dir.join("ca.crt"))
                .context("tunnel CA cert not found — run `akai-agent init` first")?;
            let wcrt = std::fs::read(cert_dir.join("worker.crt"))
                .context("tunnel worker cert not found — run `akai-agent init` first")?;
            let wkey = std::fs::read(cert_dir.join("worker.key"))
                .context("tunnel worker key not found — run `akai-agent init` first")?;
            let tc = crate::tunnel::TunnelClient::new(
                &cfg.tunnel_host,
                cfg.tunnel_port,
                &cfg.worker_id,
                cfg.rpc_port,
                ca,
                wcrt,
                wkey,
                tunnel_connected.clone(),
            );
            tokio::spawn(async move {
                if let Err(e) = tc.run().await {
                    tracing::error!("tunnel client failed: {e}");
                }
            });
        }

        let mut heartbeat_tick  = tokio::time::interval(Duration::from_secs(30));
        let mut tunnel_tick     = tokio::time::interval(Duration::from_secs(120));
        let mut update_tick     = tokio::time::interval(Duration::from_secs(86400));
        heartbeat_tick.tick().await;
        tunnel_tick.tick().await;
        update_tick.tick().await;

        // Petals child process (if model is set via heartbeat)
        let mut petals_child: Option<tokio::process::Child> = None;

        loop {
            tokio::select! {
                _ = heartbeat_tick.tick() => {
                    if use_tunnel && !tunnel_connected.load(Ordering::Relaxed) {
                        eprintln!("mTLS tunnel not connected — skipping heartbeat");
                        continue;
                    }

                    if !use_tunnel && !wireguard::check_tunnel(&cfg.wg_ip) {
                        eprintln!("WireGuard tunnel is down — pausing heartbeats until re-established");
                        match wireguard::ensure_tunnel(&cfg.wg_ip) {
                            Ok(()) => println!("WireGuard tunnel re-established — resuming"),
                            Err(e) => {
                                eprintln!("Cannot re-establish tunnel: {e}");
                                eprintln!("Workers will be unreachable. Will retry next cycle.");
                                continue;
                            }
                        }
                    }

                    if matches!(child.try_wait(), Ok(Some(_))) {
                        eprintln!("rpc-server exited — respawning...");
                        child = rpc::spawn_rpc_server(&rpc_path, cfg.rpc_port)?;
                        println!("rpc-server restarted on :{}", cfg.rpc_port);
                    }

                    let hub_reachable = true;

                    match client.heartbeat(
                        &cfg.worker_id, cfg.gpu, cfg.vram_gb, cfg.rpc_port, hub_reachable
                    ).await {
                        Ok(resp) => {
                            let tunnel_ok = if use_tunnel { tunnel_connected.load(Ordering::Relaxed) } else { true };
                            tracing::info!("Heartbeat OK (tunnel={}, hub={})", tunnel_ok, hub_reachable);

                            // Check if model changed — restart petals if needed
                            if !resp.model.is_empty() {
                                if resp.model != cfg.petals_model || petals_child.as_mut().is_none() {
                                    if !resp.model.is_empty() {
                                        eprintln!("Model set/changed: {}. Starting petals...", resp.model);
                                        // Kill existing petals process
                                        if let Some(mut child) = petals_child.take() {
                                            child.kill().await.ok();
                                            child.wait().await.ok();
                                        }
                                        // Start new petals process
                                        let args = petals::petals_args(&resp.model, 50052);
                                        let mut cmd = tokio::process::Command::new("python3");
                                        cmd.args(&args);
                                        cmd.stdout(Stdio::piped());
                                        cmd.stderr(Stdio::piped());
                                        match cmd.spawn() {
                                            Ok(child) => {
                                                petals_child = Some(child);
                                                cfg.petals_model = resp.model.clone();
                                                config::save_config(&cfg).ok();
                                                println!("Petals server started for model: {}", resp.model);
                                            }
                                            Err(e) => eprintln!("Failed to start petals: {e}"),
                                        }
                                    }
                                }
                            }

                            let my_commit = rpc::rpc_commit_hash();
                            if !resp.hub_commit.is_empty() && !my_commit.is_empty() && my_commit != resp.hub_commit {
                                eprintln!("Hub commit mismatch: hub={}, local={}. Rebuilding rpc-server...", resp.hub_commit, my_commit);
                                child.kill().ok();
                                child.wait().ok();
                                let old_binary = std::path::PathBuf::from(&cfg.rpc_binary);
                                if old_binary.exists() {
                                    let _ = std::fs::remove_file(&old_binary);
                                }
                                match rpc::ensure_rpc_server().await {
                                    Ok(new_path) => {
                                        rpc_path = new_path;
                                        if let Ok(new_cfg) = crate::config::load_config() {
                                            cfg = new_cfg;
                                        }
                                        child = rpc::spawn_rpc_server(&rpc_path, cfg.rpc_port)?;
                                        println!("rpc-server rebuilt and restarted (hub commit: {})", resp.hub_commit);
                                    }
                                    Err(e) => eprintln!("Rebuild failed: {e}"),
                                }
                            }
                        }
                        Err(e) if is_not_found(&e) => {
                            eprintln!("Not in registry — re-registering...");
                            if use_tunnel {
                                let _ = client.register(
                                    &cfg.worker_id, &cfg.worker_name,
                                    &cfg.wg_ip, &cfg.wg_peer_id,
                                    cfg.gpu, cfg.vram_gb, cfg.rpc_port,
                                    None,
                                ).await;
                            } else {
                                let wg_pub = crate::wireguard::get_wg_public_key();
                                let _ = client.register(
                                    &cfg.worker_id, &cfg.worker_name,
                                    &cfg.wg_ip, &cfg.wg_peer_id,
                                    cfg.gpu, cfg.vram_gb, cfg.rpc_port,
                                    wg_pub,
                                ).await;
                            }
                        }
                        Err(e) => eprintln!("Heartbeat failed: {e}"),
                    }
                }

                _ = tunnel_tick.tick() => {
                    if !use_tunnel && !wireguard::check_tunnel(&cfg.wg_ip) {
                        eprintln!("Periodic check: WireGuard tunnel is down");
                        match wireguard::ensure_tunnel(&cfg.wg_ip) {
                            Ok(()) => println!("WireGuard tunnel re-established"),
                            Err(e) => eprintln!("Cannot re-establish tunnel: {e}"),
                        }
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

    pub async fn start_worker(
        hub_addr: &str,
        worker_id: &str,
        model_path: &str,
        layer_offset: usize,
        num_layers: usize,
    ) -> Result<()> {
        let gpu_info = gpu::detect_gpu();

        println!("Ensuring llama-server is available...");
        let llama_server = rpc::ensure_llama_server().await
            .context("Failed to download llama-server")?;
        println!("llama-server: {}", llama_server.display());

        let n_gpu_layers = if gpu_info.has_gpu { 99 } else { 0 };

        let cfg = worker::WorkerConfig {
            hub_addr: hub_addr.to_string(),
            worker_id: worker_id.to_string(),
            has_gpu: gpu_info.has_gpu,
            vram_gb: gpu_info.vram_gb as f32,
            layer_offset,
            num_layers,
            model_path: model_path.to_string(),
            llama_server_path: Some(llama_server.to_string_lossy().to_string()),
            n_gpu_layers,
        };

        worker::run_worker(cfg).await
    }

    async fn fetch_tunnel_certs(cfg: &mut config::Config, queue_url: &str, username: &str) -> Result<()> {
        let cert_dir = config::data_dir().join("tunnel-certs");
        let ca_path = cert_dir.join("ca.crt");
        let wcrt_path = cert_dir.join("worker.crt");
        let wkey_path = cert_dir.join("worker.key");

        if ca_path.exists() && wcrt_path.exists() && wkey_path.exists() {
            println!("  Tunnel certs already exist at {}", cert_dir.display());
            if cfg.tunnel_host.is_empty() || cfg.tunnel_port == 0 {
                let client = QueueClient::new(queue_url, username);
                let certs = client.fetch_tunnel_certs().await
                    .context("Failed to fetch tunnel server info")?;
                cfg.tunnel_host = certs.tunnel_host;
                cfg.tunnel_port = certs.tunnel_port;
                config::save_config(cfg)?;
                println!("  Updated tunnel server: {}:{}", cfg.tunnel_host, cfg.tunnel_port);
            }
            return Ok(());
        }

        let client = QueueClient::new(queue_url, username);
        let certs = client.fetch_tunnel_certs().await
            .context("Failed to fetch tunnel certs")?;

        std::fs::create_dir_all(&cert_dir)
            .context("failed to create tunnel-certs directory")?;

        std::fs::write(&ca_path, &certs.ca_cert)
            .context("failed to write CA cert")?;
        std::fs::write(&wcrt_path, &certs.worker_cert)
            .context("failed to write worker cert")?;
        std::fs::write(&wkey_path, &certs.worker_key)
            .context("failed to write worker key")?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&wkey_path, std::fs::Permissions::from_mode(0x600))
                .ok();
        }

        println!("  CA:     {}", ca_path.display());
        println!("  Cert:   {}", wcrt_path.display());
        println!("  Key:    {}", wkey_path.display());
        println!("  Server: {}:{}", certs.tunnel_host, certs.tunnel_port);

        cfg.tunnel_host = certs.tunnel_host;
        cfg.tunnel_port = certs.tunnel_port;
        config::save_config(cfg)?;

        Ok(())
    }

    pub async fn install() -> Result<()> {
        crate::service::install()
    }

    pub async fn status() -> Result<()> {
        let cfg = config::load_config()
            .context("Config not found. Run `akai-agent init` first.")?;
        let client = QueueClient::from_config(&cfg);
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

    pub async fn start_petals(model: &str, port: u16, quantize: Option<String>) -> Result<()> {
        println!("Starting Petals worker for model: {}", model);
        println!("Port: {}", port);
        if let Some(ref q) = quantize {
            println!("Quantization: {}", q);
        }
        petals::run_petals_worker(model.to_string(), port, quantize).await
    }

    fn is_not_found(e: &anyhow::Error) -> bool {
        e.to_string().contains("404")
    }
}