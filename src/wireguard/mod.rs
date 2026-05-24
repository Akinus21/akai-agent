mod linux;
mod macos;
mod windows;

use anyhow::Result;
use crate::queue_client::ProvisionResponse;

pub fn get_wg_public_key() -> Option<String> {
    let conf_path = match std::env::consts::OS {
        "linux" | "macos" => "/etc/wireguard/wg0.conf",
        "windows" => return None,
        _ => return None,
    };
    let conf = std::fs::read_to_string(conf_path).ok()?;
    for line in conf.lines() {
        let line = line.trim();
        if line.starts_with("PrivateKey") {
            let key = line.split('=').nth(1)?.trim();
            let mut child = std::process::Command::new("wg")
                .args(["pubkey"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .ok()?;
            {
                use std::io::Write;
                child.stdin.as_mut()?.write_all(key.as_bytes()).ok();
            }
            let output = child.wait_with_output().ok()?;
            if output.status.success() {
                return Some(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
        }
    }
    None
}

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
        _ => {
            let name = "wg0";
            let output = std::process::Command::new("wg")
                .args(["show", name])
                .output();
            match output {
                Ok(o) if o.status.success() => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    if stdout.contains("latest handshake") {
                        return true;
                    }
                    if stdout.contains("endpoint:") && stdout.contains("transfer:") {
                        return true;
                    }
                    false
                }
                _ => false,
            }
        }
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
                .args(["down", name])
                .output();
            let output = std::process::Command::new("wg-quick")
                .args(["up", name])
                .output()?;
            if !output.status.success() {
                anyhow::bail!("wg-quick up failed: {}", String::from_utf8_lossy(&output.stderr));
            }
            Ok(())
        }
        _ => anyhow::bail!("Unsupported OS"),
    }
}