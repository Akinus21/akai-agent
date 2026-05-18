use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;
use crate::queue_client::ProvisionResponse;

pub fn configure(provision: &ProvisionResponse) -> Result<()> {
    let private_key = provision.private_key.as_ref()
        .context("private_key missing from provision response")?;
    let wg_ip = provision.wg_ip.as_ref()
        .context("wg_ip missing from provision response")?;
    let server_public_key = provision.server_public_key.as_ref()
        .context("server_public_key missing from provision response")?;
    let endpoint = provision.endpoint.as_ref()
        .context("endpoint missing from provision response")?;

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
         AllowedIPs = 10.8.0.0/24\n",
        private_key,
        wg_ip,
        provision.dns.as_deref().unwrap_or("1.1.1.1"),
        server_public_key,
        endpoint
    );

    let iface = wg_ip.rsplitn(2, '.').last().unwrap_or("1");
    let cfg_file = wg_dir.join(format!("wg{}.conf", iface));
    fs::write(&cfg_file, &config)?;

    let output = Command::new("wg-quick")
        .args(["up", &format!("wg{}", iface)])
        .output()?;

    if !output.status.success() {
        bail!("wg-quick failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    Ok(())
}