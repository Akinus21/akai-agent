pub fn install() -> anyhow::Result<()> {
    match std::env::consts::OS {
        "linux" => crate::service::systemd::install(),
        "macos" => crate::service::launchd::install(),
        "windows" => crate::service::windows::install(),
        _ => anyhow::bail!("Unsupported OS: {}", std::env::consts::OS),
    }
}