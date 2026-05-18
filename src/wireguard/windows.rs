use anyhow::Result;
use std::fs;
use std::path::Path;
use std::process::Command;
use crate::queue_client::ProvisionResponse;

pub fn configure(provision: &ProvisionResponse) -> Result<()> {
    let config_dir = std::env::var("APPDATA").unwrap_or_else(|_| ".".to_string());
    let wg_dir = Path::new(&config_dir).join("WireGuard").join("Configs");

    fs::create_dir_all(&wg_dir)?;

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
        provision.private_key,
        provision.wg_ip,
        provision.dns.as_deref().unwrap_or("1.1.1.1"),
        provision.server_public_key,
        provision.endpoint
    );

    let iface = provision.wg_ip.rsplitn(2, '.').last().unwrap_or("1");
    let cfg_file = wg_dir.join(format!("wg{}.conf", iface));
    fs::write(&cfg_file, &config)?;

    Command::new("wireguard.exe")
        .arg("/install-tunnels")
        .arg(&format!("wg{}", iface))
        .status()?;

    Ok(())
}