use anyhow::Result;
use std::process::Command;

pub fn install() -> Result<()> {
    let script = r#"$svc = New-Object -ServiceObject -Name 'Akai-Agent'
$svc.DisplayName = 'Akai-Agent GPU Worker Service'
$svc.Description = 'Remote GPU worker for akai-net'
$svc.StartType = 'Automatic'
$svc.ExecutablePath = '%INSTALLPATH%\akai-agent.exe'
$svc.Parameters = 'start'
"#;

    let ps1 = std::env::temp_dir().join("akai-agent-install.ps1");
    std::fs::write(&ps1, script)?;

    Command::new("powershell")
        .args(["-ExecutionPolicy", "Bypass", "-File", &ps1.to_string_lossy()])
        .status()?;

    std::fs::remove_file(ps1).ok();

    Ok(())
}