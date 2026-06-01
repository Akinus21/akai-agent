use anyhow::{bail, Context, Result};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering, Arc};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub struct PetalsServer {
    model_name: String,
    port: u16,
    max_length: Option<u16>,
    quantize: Option<String>,
    public_name: Option<String>,
}

impl PetalsServer {
    pub fn new(model_name: String, port: u16) -> Self {
        Self {
            model_name,
            port,
            max_length: None,
            quantize: None,
            public_name: None,
        }
    }

    pub fn max_length(mut self, len: u16) -> Self {
        self.max_length = Some(len);
        self
    }

    pub fn quantize(mut self, q: String) -> Self {
        self.quantize = Some(q);
        self
    }

    pub fn public_name(mut self, name: String) -> Self {
        self.public_name = Some(name);
        self
    }

    async fn ensure_petals_installed() -> Result<()> {
        let output = Command::new("python3")
            .args(["-m", "pip", "show", "petals"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .await?;

        if output.status.success() {
            return Ok(());
        }

        tracing::info!("Installing Petals...");
        let mut cmd = Command::new("python3");
        cmd.args(["-m", "pip", "install", "git+https://github.com/bigscience-workshop/petals"]);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let child = cmd.spawn().context("Failed to spawn pip install")?;
        let output = child.wait_with_output().await
            .context("Failed to wait for pip install")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("Petals installation failed: {}", stderr);
        }

        tracing::info!("Petals installed successfully");
        Ok(())
    }

    pub async fn run(&self) -> Result<()> {
        Self::ensure_petals_installed().await?;

        let mut args = vec![
            "-m".to_string(), "petals.cli.run_server".to_string(),
            self.model_name.clone(),
            "--port".to_string(),
            self.port.to_string(),
        ];

        if let Some(max_len) = self.max_length {
            args.push("--max-length".to_string());
            args.push(max_len.to_string());
        }

        if let Some(ref q) = self.quantize {
            args.push("--quantize".to_string());
            args.push(q.clone());
        }

        if let Some(ref name) = self.public_name {
            args.push("--public_name".to_string());
            args.push(name.clone());
        }

        tracing::info!("Starting Petals server: python3 {}", args.join(" "));

        let mut cmd = Command::new("python3");
        cmd.args(&args);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().context("Failed to spawn petals server")?;

        let stdout = child.stdout.take()
            .expect("stdout not captured");
        let mut reader = BufReader::new(stdout).lines();

        let started = Arc::new(AtomicBool::new(false));
        let started_clone = started.clone();

        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::info!("[petals] {}", line);
                if line.contains("Running on") || line.contains("Server started") {
                    started_clone.store(true, Ordering::SeqCst);
                }
            }
        });

        let timeout = Duration::from_secs(120);
        let start = Instant::now();
        while !started.load(Ordering::SeqCst) {
            if start.elapsed() > timeout {
                child.kill().await.ok();
                bail!("Petals server failed to start within 120s");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        tracing::info!("Petals server ready on port {}", self.port);
        child.wait().await?;
        Ok(())
    }
}

pub async fn run_petals_worker(
    model_name: String,
    port: u16,
    quantize: Option<String>,
) -> Result<()> {
    let mut server = PetalsServer::new(model_name, port);
    if let Some(q) = quantize {
        server = server.quantize(q);
    }
    server.run().await
}

pub fn spawn_petals(model: &str, port: u16) -> Result<tokio::process::Child> {
    let mut cmd = Command::new("python3");
    cmd.args(&[
        "-m", "petals.cli.run_server",
        model,
        "--port", &port.to_string(),
    ]);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let child = cmd.spawn().context("Failed to spawn petals server")?;
    Ok(child)
}

pub fn petals_args(model: &str, port: u16) -> Vec<String> {
    vec![
        "-m".to_string(), "petals.cli.run_server".to_string(),
        model.to_string(),
        "--port".to_string(),
        port.to_string(),
    ]
}

pub async fn check_petals_health(port: u16) -> Result<bool> {
    let url = format!("http://127.0.0.1:{}/health", port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    match client.get(&url).send().await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(_) => Ok(false),
    }
}