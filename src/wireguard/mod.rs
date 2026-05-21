mod linux;
mod macos;
mod windows;

use anyhow::Result;
use crate::queue_client::ProvisionResponse;

pub async fn configure(provision: &ProvisionResponse) -> Result<()> {
    match std::env::consts::OS {
        "linux" => linux::configure(provision),
        "macos" => macos::configure(provision),
        "windows" => windows::configure(provision),
        _ => anyhow::bail!("Unsupported OS: {}", std::env::consts::OS),
    }
}

pub fn check_tunnel(wg_ip: &str) -> bool {
    match std::env::consts::OS {
        "linux" => linux::check_tunnel(wg_ip),
        "macos" => {
            let name = "wg0";
            let output = std::process::Command::new("wg")
                .args(["show", &name])
                .output();
            match output {
                Ok(o) if o.status.success() => {
                    String::from_utf8_lossy(&o.stdout).contains("latest handshake")
                }
                _ => false,
            }
        }
        "windows" => {
            let name = "wg0";
            let output = std::process::Command::new("wg")
                .args(["show", &name])
                .output();
            match output {
                Ok(o) if o.status.success() => {
                    String::from_utf8_lossy(&o.stdout).contains("latest handshake")
                }
                _ => false,
            }
        }
        _ => false,
    }
}

pub fn ensure_tunnel(wg_ip: &str) -> Result<()> {
    match std::env::consts::OS {
        "linux" => linux::ensure_tunnel(wg_ip),
        "macos" | "windows" => {
            if check_tunnel(wg_ip) {
                return Ok(());
            }
            let name = "wg0";
            eprintln!("WireGuard tunnel is down — attempting to re-establish...");
            let _ = std::process::Command::new("wg-quick")
                .args(["down", &name])
                .output();
            let output = std::process::Command::new("wg-quick")
                .args(["up", &name])
                .output()?;
            if !output.status.success() {
                anyhow::bail!("wg-quick up failed: {}", String::from_utf8_lossy(&output.stderr));
            }
            Ok(())
        }
        _ => anyhow::bail!("Unsupported OS"),
    }
}