use anyhow::Result;
use crate::queue_client::ProvisionResponse;

pub async fn configure(provision: &ProvisionResponse) -> Result<()> {
    match std::env::consts::OS {
        "linux" => crate::wireguard::linux::configure(provision),
        "macos" => crate::wireguard::macos::configure(provision),
        "windows" => crate::wireguard::windows::configure(provision),
        _ => anyhow::bail!("Unsupported OS: {}", std::env::consts::OS),
    }
}