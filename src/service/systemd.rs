use anyhow::Result;
use std::fs;

pub fn install() -> Result<()> {
    let service = r#"[Unit]
Description=Akai-Agent GPU Worker Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/akai-agent start
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
"#;

    fs::write("/etc/systemd/system/akai-agent.service", service)?;
    std::process::Command::new("systemctl")
        .args(["daemon-reload"])
        .status()?;
    std::process::Command::new("systemctl")
        .args(["enable", "akai-agent"])
        .status()?;
    std::process::Command::new("systemctl")
        .args(["start", "akai-agent"])
        .status()?;

    Ok(())
}