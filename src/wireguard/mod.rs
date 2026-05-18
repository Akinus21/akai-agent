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