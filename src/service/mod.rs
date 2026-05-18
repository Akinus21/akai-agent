mod systemd;
mod launchd;
mod windows;

pub fn install() -> anyhow::Result<()> {
    match std::env::consts::OS {
        "linux" => systemd::install(),
        "macos" => launchd::install(),
        "windows" => windows::install(),
        _ => anyhow::bail!("Unsupported OS: {}", std::env::consts::OS),
    }
}