use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use crate::queue_client::ProvisionResponse;

fn iface_name(_wg_ip: &str) -> &'static str {
    "wg0"
}

fn is_ostree() -> bool {
    Path::new("/ostree").exists()
        || Command::new("which")
            .arg("rpm-ostree")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

fn is_bluefin() -> bool {
    if let Ok(contents) = std::fs::read_to_string("/etc/os-release") {
        contents.to_lowercase().contains("bluefin")
    } else {
        false
    }
}

fn use_manual_wg() -> bool {
    (is_ostree() && !is_container()) || is_bluefin()
}

fn is_container() -> bool {
    Path::new("/run/.containerenv").exists()
        || Path::new("/.dockerenv").exists()
        || std::fs::read_to_string("/proc/1/cgroup")
            .map(|c| c.contains("docker") || c.contains("lxc") || c.contains("distrobox"))
            .unwrap_or(false)
}

fn can_sudo() -> bool {
    Command::new("sudo")
        .args(["-n", "true"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn has_wg_quick() -> bool {
    Command::new("which")
        .arg("wg-quick")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn has_wg() -> bool {
    Command::new("which")
        .arg("wg")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn configure(provision: &ProvisionResponse) -> Result<()> {
    let private_key = provision.private_key.as_ref()
        .context("private_key missing from provision response")?;
    let wg_ip = provision.wg_ip.as_ref()
        .context("wg_ip missing from provision response")?;
    let server_public_key = provision.server_public_key.as_ref()
        .context("server_public_key missing from provision response")?;
    let endpoint = provision.endpoint.as_ref()
        .context("endpoint missing from provision response")?;

    let name = iface_name(wg_ip);

    write_config(name, private_key, wg_ip, server_public_key, endpoint, provision.dns.as_deref(), provision.preshared_key.as_deref())?;

    if use_manual_wg() {
        bring_up_manual(name, wg_ip, server_public_key, endpoint, provision.preshared_key.as_deref())?;
    } else {
        let _ = Command::new("sudo")
            .args(["wg-quick", "down", name])
            .output();
        let output = Command::new("sudo")
            .args(["wg-quick", "up", name])
            .output()?;
        if !output.status.success() {
            bail!("wg-quick failed: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    Ok(())
}

fn write_config(
    name: &str, private_key: &str, wg_ip: &str,
    server_public_key: &str, endpoint: &str, dns: Option<&str>,
    preshared_key: Option<&str>,
) -> Result<()> {
    let peer_section = if let Some(psk) = preshared_key {
        format!(
            "[Peer]\n\
             PublicKey = {}\n\
             PresharedKey = {}\n\
             Endpoint = {}\n\
             AllowedIPs = 10.8.0.0/24\n\
             PersistentKeepalive = 25\n",
            server_public_key, psk, endpoint
        )
    } else {
        format!(
            "[Peer]\n\
             PublicKey = {}\n\
             Endpoint = {}\n\
             AllowedIPs = 10.8.0.0/24\n\
             PersistentKeepalive = 25\n",
            server_public_key, endpoint
        )
    };
    let config = format!(
        "[Interface]\n\
         PrivateKey = {}\n\
         Address = {}/24\n\
         DNS = {}\n\
         \n\
         {}",
        private_key,
        wg_ip,
        dns.unwrap_or("1.1.1.1"),
        peer_section
    );

    let wg_dir = Path::new("/etc/wireguard");
    if !wg_dir.exists() {
        if can_sudo() {
            let wg_dir_s = wg_dir.to_string_lossy();
            let _ = Command::new("sudo")
                .args(["mkdir", "-p", &*wg_dir_s])
                .status();
        }
        if !wg_dir.exists() {
            fs::create_dir_all(wg_dir)?;
        }
    }

    let cfg_file = wg_dir.join(format!("{}.conf", name));
    if can_sudo() {
        let tmp = format!("/tmp/{}.conf", name);
        fs::write(&tmp, &config)?;
        let cfg_file_s = cfg_file.to_string_lossy();
        let status = Command::new("sudo")
            .args(["cp", &*tmp, &*cfg_file_s])
            .status()?;
        let _ = Command::new("rm").arg(&tmp).status();
        if !status.success() {
            bail!("Failed to write WireGuard config to {}", cfg_file.display());
        }
    } else {
        fs::write(&cfg_file, config)?;
    }

    Ok(())
}

fn bring_up_manual(name: &str, wg_ip: &str, server_public_key: &str, endpoint: &str, preshared_key: Option<&str>) -> Result<()> {
    let _ = Command::new("sudo")
        .args(["ip", "link", "del", name])
        .output();

    let status = Command::new("sudo")
        .args(["ip", "link", "add", "dev", name, "type", "wireguard"])
        .status()?;
    if !status.success() {
        bail!("Failed to create WireGuard interface {}", name);
    }

    let tmp_key = format!("/tmp/{}_privatekey", name);
    let conf = fs::read_to_string(format!("/etc/wireguard/{}.conf", name))?;
    let pk = conf.lines()
        .find(|l| l.trim().starts_with("PrivateKey"))
        .and_then(|l| l.splitn(2, '=').nth(1))
        .map(|v| v.trim())
        .unwrap_or("");
    fs::write(&tmp_key, pk)?;

    let mut cmd = Command::new("sudo");
    cmd.arg("wg").arg("set").arg(name).arg("private-key").arg(&tmp_key)
        .arg("peer").arg(server_public_key)
        .arg("endpoint").arg(endpoint)
        .arg("allowed-ips").arg("10.8.0.0/24")
        .arg("persistent-keepalive").arg("25");

    let tmp_psk = if let Some(psk) = preshared_key {
        let path = format!("/tmp/{}_presharedkey", name);
        fs::write(&path, psk)?;
        cmd.arg("preshared-key").arg(&path);
        Some(path)
    } else {
        None
    };

    let status = cmd.status()?;
    let _ = Command::new("sudo").arg("rm").arg("-f").arg(&tmp_key).status();
    if let Some(ref psk_path) = tmp_psk {
        let _ = Command::new("sudo").arg("rm").arg("-f").arg(psk_path).status();
    }
    if !status.success() {
        bail!("Failed to configure WireGuard interface {}", name);
    }

    Command::new("sudo")
        .args(["ip", "address", "add", &format!("{}/24", wg_ip), "dev", name])
        .status()
        .context("Failed to assign IP to WireGuard interface")?;

    Command::new("sudo")
        .args(["ip", "link", "set", name, "up"])
        .status()
        .context("Failed to bring up WireGuard interface")?;

    println!("  WireGuard interface {} up (manual config)", name);
    Ok(())
}

pub fn check_tunnel(wg_ip: &str) -> bool {
    let name = iface_name(wg_ip);

    let try_wg = |cmd: &mut std::process::Command| -> bool {
        match cmd.args(["show", &name]).output() {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                if stdout.contains("latest handshake") {
                    return true;
                }
                if stdout.contains("peer:") && stdout.contains("endpoint:") {
                    return true;
                }
                if stdout.contains(&format!("allowed ips: {}/32", wg_ip)) {
                    if stdout.contains("transfer:") {
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    };

    if try_wg(&mut Command::new("sudo").arg("wg")) {
        return true;
    }
    try_wg(&mut Command::new("wg"))
}

pub fn ensure_tunnel(wg_ip: &str) -> Result<()> {
    if check_tunnel(wg_ip) {
        return Ok(());
    }

    let name = iface_name(wg_ip);
    let conf_path = format!("/etc/wireguard/{}.conf", name);
    eprintln!("WireGuard tunnel is down — attempting to re-establish...");

    let conf = if Path::new(&conf_path).exists() {
        Some(fs::read_to_string(&conf_path)?)
    } else {
        None
    };

    let parse_conf = |c: &str| -> (String, String, String, Option<String>) {
        let pk = c.lines()
            .find(|l| l.trim().starts_with("PrivateKey"))
            .and_then(|l| l.splitn(2, '=').nth(1))
            .map(|v| v.trim().to_string())
            .unwrap_or_default();
        let spk = c.lines()
            .find(|l| l.trim().starts_with("PublicKey"))
            .and_then(|l| l.splitn(2, '=').nth(1))
            .map(|v| v.trim().to_string())
            .unwrap_or_default();
        let ep = c.lines()
            .find(|l| l.trim().starts_with("Endpoint"))
            .and_then(|l| l.splitn(2, '=').nth(1))
            .map(|v| v.trim().to_string())
            .unwrap_or_default();
        let psk = c.lines()
            .find(|l| l.trim().starts_with("PresharedKey"))
            .and_then(|l| l.splitn(2, '=').nth(1))
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        (pk, spk, ep, psk)
    };

    if use_manual_wg() {
        if let Some(ref c) = conf {
            let (_, server_public_key, endpoint, preshared_key) = parse_conf(c);
            if !server_public_key.is_empty() && !endpoint.is_empty() {
                let _ = Command::new("sudo")
                    .args(["ip", "link", "del", name])
                    .output();
                bring_up_manual(name, wg_ip, &server_public_key, &endpoint, preshared_key.as_deref())?;
                println!("WireGuard tunnel configured (manual)");
                return Ok(());
            }
        }
        bail!("WireGuard config missing PublicKey or Endpoint.");
    }

    if has_wg_quick() {
        let _ = Command::new("sudo")
            .args(["wg-quick", "down", name])
            .output();
        let output = Command::new("sudo")
            .args(["wg-quick", "up", name])
            .output()?;
        if output.status.success() {
            println!("WireGuard tunnel configured (wg-quick)");
            return Ok(());
        }
        eprintln!("wg-quick up failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    if has_wg() {
        if let Some(ref c) = conf {
            let (_, server_public_key, endpoint, preshared_key) = parse_conf(c);
            if !server_public_key.is_empty() && !endpoint.is_empty() {
                let _ = Command::new("sudo")
                    .args(["ip", "link", "del", name])
                    .output();
                bring_up_manual(name, wg_ip, &server_public_key, &endpoint, preshared_key.as_deref())?;
                println!("WireGuard tunnel configured (manual fallback)");
                return Ok(());
            }
        }
    }

    bail!("No WireGuard configuration found at {}.", conf_path)
}