mod systemd;
mod launchd;
mod windows;

pub fn install_service() -> anyhow::Result<()> {
    match std::env::consts::OS {
        "linux" => systemd::install(),
        "macos" => launchd::install(),
        "windows" => windows::install(),
        _ => anyhow::bail!("Unsupported OS: {}", std::env::consts::OS),
    }
}

pub fn uninstall_service() -> anyhow::Result<()> {
    match std::env::consts::OS {
        "linux" => systemd::uninstall(),
        "macos" => Ok(()),
        "windows" => Ok(()),
        _ => anyhow::bail!("Unsupported OS: {}", std::env::consts::OS),
    }
}