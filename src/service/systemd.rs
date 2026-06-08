use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;

fn is_ostree() -> bool {
    Path::new("/ostree").exists()
        || Command::new("which")
            .arg("rpm-ostree")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

fn is_container() -> bool {
    Path::new("/run/.containerenv").exists()
        || Path::new("/.dockerenv").exists()
        || std::fs::read_to_string("/proc/1/cgroup")
            .map(|c| c.contains("docker") || c.contains("lxc") || c.contains("distrobox"))
            .unwrap_or(false)
}

const ATOMIC_INSTALL_DIR: &str = "/var/lib/akai-agent/bin";
const ATOMIC_DATA_DIR: &str = "/var/lib/akai-agent/data";

fn find_akai_agent() -> Option<String> {
    let search_paths: Vec<String> = if is_ostree() {
        vec![
            "/usr/local/bin/akai-agent".to_string(),
            "/usr/bin/akai-agent".to_string(),
            format!("{}/akai-agent", ATOMIC_INSTALL_DIR),
        ]
    } else {
        vec![
            "/usr/local/bin/akai-agent".to_string(),
            "/usr/bin/akai-agent".to_string(),
        ]
    };

    for path in search_paths {
        if Path::new(&path).exists() {
            return Some(path);
        }
    }

    std::env::var("PATH").unwrap_or_default()
        .split(':')
        .filter_map(|p| {
            let full = format!("{}/akai-agent", p);
            if Path::new(&full).exists() {
                Some(full)
            } else {
                None
            }
        })
        .next()
}

fn install_binary_for_atomic() -> Result<String> {
    if !is_ostree() || is_container() {
        return Ok(find_akai_agent().unwrap_or_else(|| "/usr/local/bin/akai-agent".to_string()));
    }

    let current_binary = find_akai_agent().context("akai-agent not found in PATH. Install it first: cargo install akai-agent")?;

    if current_binary.starts_with(ATOMIC_INSTALL_DIR) {
        return Ok(current_binary);
    }

    println!("Atomic distro detected. Installing binary to {}...", ATOMIC_INSTALL_DIR);

    let dest_dir = Path::new(ATOMIC_INSTALL_DIR);
    fs::create_dir_all(dest_dir)?;

    let dest = dest_dir.join("akai-agent");
    fs::copy(&current_binary, &dest).context("Failed to copy binary")?;

    let self_binary = std::env::current_exe()
        .context("Failed to get current executable path")?;

    if self_binary.exists() && self_binary != dest {
        fs::copy(&self_binary, &dest).context("Failed to copy self to install dir")?;
    }

    println!("  Binary installed to {}", dest.display());

    Ok(dest.to_string_lossy().to_string())
}

fn write_service_file(binary_path: &str) -> Result<()> {
    let service = format!(r#"[Unit]
Description=Akai-Agent GPU Worker Service
After=network.target

[Service]
Type=simple
ExecStart={binary_path} start
Restart=always
RestartSec=10
User=root
Environment="DATA_DIR={data_dir}"

[Install]
WantedBy=multi-user.target
"#,
        binary_path = binary_path,
        data_dir = ATOMIC_DATA_DIR
    );

    fs::write("/etc/systemd/system/akai-agent.service", service)?;
    println!("  Service file written to /etc/systemd/system/akai-agent.service");
    Ok(())
}

pub fn install() -> Result<()> {
    if is_ostree() && !is_container() {
        if !Path::new("/var").exists() {
            bail!("/var not found - cannot install on this atomic distro");
        }
    }

    let binary_path = if is_ostree() && !is_container() {
        install_binary_for_atomic()?
    } else {
        find_akai_agent().unwrap_or_else(|| "/usr/local/bin/akai-agent".to_string())
    };

    if !Path::new(&binary_path).exists() {
        bail!(
            "akai-agent binary not found at {}. Run: cargo install akai-agent",
            binary_path
        );
    }

    println!("Installing akai-agent systemd service...");
    println!("  Binary: {}", binary_path);

    write_service_file(&binary_path)?;

    println!("  Running systemctl daemon-reload...");
    let status = Command::new("systemctl")
        .args(["daemon-reload"])
        .status()
        .context("systemctl daemon-reload failed")?;
    if !status.success() {
        eprintln!("  Warning: systemctl daemon-reload returned non-zero");
    }

    println!("  Enabling akai-agent service...");
    let status = Command::new("systemctl")
        .args(["enable", "akai-agent"])
        .status()
        .context("systemctl enable failed")?;
    if !status.success() {
        bail!("systemctl enable akai-agent failed");
    }

    println!("  Starting akai-agent service...");
    let status = Command::new("systemctl")
        .args(["start", "akai-agent"])
        .status()
        .context("systemctl start failed")?;
    if !status.success() {
        eprintln!("  Warning: systemctl start returned non-zero (may already be running)");
    }

    println!("\nService installed and started!");
    println!("  View logs: journalctl -u akai-agent -f");
    println!("  Stop service: systemctl stop akai-agent");
    println!("  Disable service: systemctl disable akai-agent");

    Ok(())
}

pub fn uninstall() -> Result<()> {
    println!("Uninstalling akai-agent systemd service...");

    let _ = Command::new("systemctl")
        .args(["stop", "akai-agent"])
        .output();

    let _ = Command::new("systemctl")
        .args(["disable", "akai-agent"])
        .output();

    if Path::new("/etc/systemd/system/akai-agent.service").exists() {
        fs::remove_file("/etc/systemd/system/akai-agent.service")?;
        println!("  Removed /etc/systemd/system/akai-agent.service");
    }

    Command::new("systemctl")
        .args(["daemon-reload"])
        .status()?;

    if is_ostree() && !is_container() {
        let bin_dir = Path::new(ATOMIC_INSTALL_DIR);
        if bin_dir.exists() {
            fs::remove_dir_all(bin_dir)?;
            println!("  Removed {}", ATOMIC_INSTALL_DIR);
        }
    }

    println!("Uninstall complete.");
    Ok(())
}