use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn install() -> Result<()> {
    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.akinus21.akai-agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/akai-agent</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
"#;

    let dir = Path::new("/Library/LaunchDaemons");
    std::fs::create_dir_all(dir)?;
    let plist_path = dir.join("com.akinus21.akai-agent.plist");
    fs::write(&plist_path, plist)?;

    std::process::Command::new("launchctl")
        .args(["load", &plist_path.to_string_lossy()])
        .status()?;

    Ok(())
}