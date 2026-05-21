use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use crate::queue_client::ProvisionResponse;

fn iface_name(wg_ip: &str) -> String {
    let iface = wg_ip.rsplitn(2, '.').last().unwrap_or("1");
    format!("wg{}", iface)
}

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

    if is_ostree() && !is_container() && !has_wg_quick() {
        configure_atomic(&name, private_key, wg_ip, server_public_key, endpoint, provision.dns.as_deref())?;
    } else {
        configure_standard(&name, private_key, wg_ip, server_public_key, endpoint, provision.dns.as_deref())?;
    }

    Ok(())
}

fn configure_standard(
    name: &str, private_key: &str, wg_ip: &str,
    server_public_key: &str, endpoint: &str, dns: Option<&str>,
) -> Result<()> {
    let wg_dir = Path::new("/etc/wireguard");
    fs::create_dir_all(wg_dir)?;

    let config = format!(
        "[Interface]\n\
         PrivateKey = {}\n\
         Address = {}/24\n\
         DNS = {}\n\
         \n\
         [Peer]\n\
         PublicKey = {}\n\
         Endpoint = {}\n\
         AllowedIPs = 10.8.0.0/24\n\
         PersistentKeepalive = 25\n",
        private_key,
        wg_ip,
        dns.unwrap_or("1.1.1.1"),
        server_public_key,
        endpoint
    );

    let cfg_file = wg_dir.join(format!("{}.conf", name));
    fs::write(&cfg_file, &config)?;

    let _ = Command::new("wg-quick")
        .args(["down", name])
        .output();

    let output = Command::new("wg-quick")
        .args(["up", name])
        .output()?;

    if !output.status.success() {
        bail!("wg-quick failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(())
}

fn configure_atomic(
    name: &str, private_key: &str, wg_ip: &str,
    server_public_key: &str, endpoint: &str, dns: Option<&str>,
) -> Result<()> {
    println!("  Atomic/immutable distro detected — configuring WireGuard manually");

    let config = format!(
        "[Interface]\n\
         PrivateKey = {}\n\
         Address = {}/24\n\
         DNS = {}\n\
         \n\
         [Peer]\n\
         PublicKey = {}\n\
         Endpoint = {}\n\
         AllowedIPs = 10.8.0.0/24\n\
         PersistentKeepalive = 25\n",
        private_key,
        wg_ip,
        dns.unwrap_or("1.1.1.1"),
        server_public_key,
        endpoint
    );

    let wg_dir = Path::new("/etc/wireguard");
    if !wg_dir.exists() && can_sudo() {
        let _ = Command::new("sudo")
            .args(["mkdir", "-p", &wg_dir.to_string_lossy()])
            .status();
    }
    if !wg_dir.exists() {
        fs::create_dir_all(wg_dir)?;
    }

    let cfg_file = wg_dir.join(format!("{}.conf", name));
    if can_sudo() {
        let tmp = format!("/tmp/{}.conf", name);
        fs::write(&tmp, &config)?;
        let status = Command::new("sudo")
            .args(["cp", &tmp, &cfg_file.to_string_lossy()])
            .status()?;
        if !status.success() {
            bail!("Failed to write WireGuard config to {}", cfg_file.display());
        }
        let _ = Command::new("rm").arg(&tmp).status();
    } else {
        fs::write(&cfg_file, &config)?;
    }

    if has_wg() && can_sudo() {
        bring_up_manual(name, wg_ip, server_public_key, endpoint)?;
        return Ok(());
    }

    if can_sudo() {
        println!("  Installing WireGuard tools via rpm-ostree (may require reboot)...");
        let status = Command::new("sudo")
            .args(["rpm-ostree", "install", "-y", "wireguard-tools"])
            .status()?;
        if status.success() {
            println!("  WireGuard tools installed. A reboot may be required.");
            println!("  After reboot, run: akai-agent start");
            bring_up_manual(name, wg_ip, server_public_key, endpoint)?;
            return Ok(());
        }
        eprintln!("  rpm-ostree install failed, trying alternative methods...");
    }

    if Path::new("/usr/bin/distrobox").exists() || Command::new("which").arg("distrobox").output().map(|o| o.status.success()).unwrap_or(false) {
        println!("  Found distrobox — using container for WireGuard...");
        configure_via_distrobox(name, &config)?;
        return Ok(());
    }

    bail!(
        "Cannot set up WireGuard on this atomic distro.\n\
         Options:\n\
         1. Install wireguard-tools: sudo rpm-ostree install wireguard-tools (reboot required)\n\
         2. Install distrobox for container-based setup\n\
         3. Run akai-agent inside a distrobox container"
    );
}

fn bring_up_manual(name: &str, wg_ip: &str, server_public_key: &str, endpoint: &str) -> Result<()> {
    let _ = Command::new("sudo")
        .args(["ip", "link", "del", name])
        .output();

    let status = Command::new("sudo")
        .args(["ip", "link", "add", "dev", name, "type", "wireguard"])
        .status()?;
    if !status.success() {
        bail!("Failed to create WireGuard interface {}", name);
    }

    let status = Command::new("sudo")
        .args(["wg", "set", name,
               "private-key", "/etc/wireguard/{}.conf"])
        .status();
    let status = if status.as_ref().is_err() || !status.as_ref().unwrap().success() {
        let tmp_key = format!("/tmp/{}_privatekey", name);
        let conf = fs::read_to_string(format!("/etc/wireguard/{}.conf", name))?;
        let pk = conf.lines()
            .find(|l| l.trim().starts_with("PrivateKey"))
            .and_then(|l| l.split('=').nth(1))
            .map(|v| v.trim())
            .unwrap_or("");
        fs::write(&tmp_key, pk)?;
        let s = Command::new("sudo")
            .args(["wg", "set", name, "private-key", &tmp_key,
                   "peer", server_public_key,
                   "endpoint", endpoint,
                   "allowed-ips", "10.8.0.0/24",
                   "persistent-keepalive", "25"])
            .status()?;
        let _ = Command::new("rm").arg(&tmp_key).status();
        s
    } else {
        status.unwrap()
    };
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

fn configure_via_distrobox(name: &str, config: &str) -> Result<()> {
    let container_name = "akai-wg";
    let list_output = Command::new("distrobox")
        .args(["list", "--no-header"])
        .output()?;
    let listing = String::from_utf8_lossy(&list_output.stdout);

    if !listing.lines().any(|l| l.contains(container_name)) {
        println!("  Creating distrobox container for WireGuard...");
        let status = Command::new("distrobox")
            .args(["create", "--name", container_name, "--image", "ubuntu:24.04", "--yes"])
            .status()?;
        if !status.success() {
            bail!("Failed to create distrobox container for WireGuard");
        }

        let install_cmd = "sudo apt-get update -qq && sudo apt-get install -y wireguard-tools iproute2";
        let status = Command::new("distrobox")
            .args(["enter", container_name, "--", "sh", "-c", install_cmd])
            .status()?;
        if !status.success() {
            bail!("Failed to install wireguard-tools in distrobox container");
        }
    }

    let tmp_conf = format!("/tmp/{}.conf", name);
    fs::write(&tmp_conf, config)?;

    let enter_cmd = format!(
        "sudo mkdir -p /etc/wireguard && \
         sudo cp /run/host/{}/{}_conf /etc/wireguard/{}.conf && \
         sudo wg-quick down {} 2>/dev/null; \
         sudo wg-quick up {}",
        tmp_conf, name, name, name, name
    );

    println!("  Starting WireGuard via distrobox...");
    let status = Command::new("distrobox")
        .args(["enter", container_name, "--", "sh", "-c", &enter_cmd])
        .status()?;

    let _ = Command::new("rm").arg(&tmp_conf).status();

    if !status.success() {
        bail!("Failed to start WireGuard via distrobox");
    }

    println!("  WireGuard running via distrobox container");
    Ok(())
}

pub fn check_tunnel(wg_ip: &str) -> bool {
    let name = iface_name(wg_ip);

    let output = match Command::new("wg").args(["show", &name]).output() {
        Ok(o) => o,
        Err(_) => return false,
    };

    if !output.status.success() {
        return false;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.contains("latest handshake")
}

pub fn ensure_tunnel(wg_ip: &str) -> Result<()> {
    if check_tunnel(wg_ip) {
        return Ok(());
    }

    let name = iface_name(wg_ip);
    eprintln!("WireGuard tunnel is down — attempting to re-establish...");

    if has_wg_quick() {
        let _ = Command::new("sudo")
            .args(["wg-quick", "down", &name])
            .output();

        let output = Command::new("sudo")
            .args(["wg-quick", "up", &name])
            .output()?;

        if output.status.success() {
            let mut waited = 0u64;
            while waited < 15 {
                if check_tunnel(wg_ip) {
                    println!("WireGuard tunnel re-established");
                    return Ok(());
                }
                std::thread::sleep(Duration::from_secs(1));
                waited += 1;
            }
        }
        eprintln!("wg-quick up failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    if has_wg() {
        let conf_path = format!("/etc/wireguard/{}.conf", name);
        if Path::new(&conf_path).exists() {
            let _ = Command::new("sudo")
                .args(["ip", "link", "del", &name])
                .output();

            let conf = fs::read_to_string(&conf_path)?;
            let private_key = conf.lines()
                .find(|l| l.trim().starts_with("PrivateKey"))
                .and_then(|l| l.split('=').nth(1))
                .map(|v| v.trim().to_string())
                .unwrap_or_default();

            let server_public_key = conf.lines()
                .find(|l| l.trim().starts_with("PublicKey"))
                .and_then(|l| l.split('=').nth(1))
                .map(|v| v.trim().to_string())
                .unwrap_or_default();

            let endpoint = conf.lines()
                .find(|l| l.trim().starts_with("Endpoint"))
                .and_then(|l| l.split('=').nth(1))
                .map(|v| v.trim().to_string())
                .unwrap_or_default();

            if !private_key.is_empty() && !server_public_key.is_empty() && !endpoint.is_empty() {
                bring_up_manual(&name, wg_ip, &server_public_key, &endpoint)?;
                let mut waited = 0u64;
                while waited < 15 {
                    if check_tunnel(wg_ip) {
                        println!("WireGuard tunnel re-established (manual)");
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_secs(1));
                    waited += 1;
                }
            }
        }
    }

    bail!("Failed to re-establish WireGuard tunnel for {}. Check WireGuard config and network connectivity.", name)
}