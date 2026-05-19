use anyhow::Result;
use std::fs;


fn find_akai_agent() -> String {
    std::env::var("PATH").unwrap_or_default()
        .split(':')
        .filter_map(|p| {
            let full = format!("{}/akai-agent", p);
            if std::path::Path::new(&full).exists() {
                Some(full)
            } else {
                None
            }
        })
        .next()
        .unwrap_or_else(|| "/usr/local/bin/akai-agent".to_string())
}

pub fn install() -> Result<()> {
    let binary_path = find_akai_agent();
    let service = format!(r#"[Unit]
Description=Akai-Agent GPU Worker Service
After=network.target

[Service]
Type=simple
ExecStart={binary_path} start
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
"#);

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