use anyhow::{anyhow, bail, Result};
use std::path::{Path, PathBuf};
use flate2::read::GzDecoder;

const GITHUB_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
const USER_AGENT: &str = concat!("akai-agent/", env!("CARGO_PKG_VERSION"));

pub fn binary_name() -> &'static str {
    if cfg!(windows) { "rpc-server.exe" } else { "rpc-server" }
}

pub fn rpc_binary_path() -> PathBuf {
    crate::config::data_dir().join(binary_name())
}

fn asset_pattern() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux",   "x86_64")  => "llama-*-bin-ubuntu-x64.tar.gz",
        ("linux",   "aarch64") => "llama-*-bin-ubuntu-arm64.tar.gz",
        ("macos",   "aarch64") => "llama-*-bin-macos-arm64.tar.gz",
        ("macos",   "x86_64")  => "llama-*-bin-macos-x64.tar.gz",
        ("windows", "x86_64")  => "llama-*-bin-win-cuda-12.4-x64.zip",
        (os, arch)             => panic!("Unsupported platform: {os}/{arch}"),
    }
}

fn glob_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() { continue; }
        if i == 0 {
            if !name.starts_with(part) { return false; }
            pos = part.len();
        } else {
            match name[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None      => return false,
            }
        }
    }
    true
}

async fn fetch_latest_release() -> Result<serde_json::Value> {
    let client = reqwest::Client::new();
    let resp = client.get(GITHUB_API)
        .header("User-Agent", USER_AGENT)
        .send().await?;
    if !resp.status().is_success() {
        bail!("GitHub API returned {}", resp.status());
    }
    Ok(resp.json().await?)
}

pub async fn needs_update(current_version: &str) -> Result<bool> {
    let release = fetch_latest_release().await?;
    let latest = release["tag_name"].as_str()
        .ok_or_else(|| anyhow!("tag_name missing from GitHub response"))?;
    Ok(latest != current_version)
}

pub fn current_version() -> String {
    crate::config::load_config()
        .map(|c| c.rpc_version)
        .unwrap_or_default()
}

fn lib_files_valid(lib_dir: &Path) -> bool {
    std::fs::read_dir(lib_dir)
        .ok()
        .map(|entries| entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".so"))
            .any(|e| e.path().metadata()
                .map(|m| m.is_file() && m.len() > 1024)
                .unwrap_or(false)))
        .unwrap_or(false)
}

pub async fn ensure_rpc_server() -> Result<PathBuf> {
    let path = rpc_binary_path();
    let lib_dir = crate::config::data_dir().join("lib");
    let libs_valid = lib_dir.exists() && lib_files_valid(&lib_dir);

    if !path.exists() || !libs_valid {
        if lib_dir.exists() {
            std::fs::remove_dir_all(&lib_dir).ok();
        }
        std::fs::remove_file(&path).ok();
        
        #[cfg(target_os = "linux")]
        if crate::build::needs_source_build() && crate::build::has_build_tools() {
            println!("NVIDIA GPU detected — building rpc-server from source with CUDA...");
            match crate::build::build_from_source() {
                Ok(p) => return Ok(p),
                Err(e) => eprintln!("Source build failed: {}. Falling back to download.", e),
            }
        }
        
        download_latest().await?;
    }
    Ok(path)
}

pub async fn download_latest() -> Result<()> {
    let dest = rpc_binary_path();
    let release = fetch_latest_release().await?;
    let tag = release["tag_name"].as_str()
        .ok_or_else(|| anyhow!("tag_name missing"))?.to_string();
    let pattern = asset_pattern();

    let assets = release["assets"].as_array()
        .ok_or_else(|| anyhow!("no assets in release"))?;
    let asset = assets.iter()
        .find(|a| {
            a["name"].as_str()
                .map(|n| glob_match(pattern, n))
                .unwrap_or(false)
        })
        .ok_or_else(|| anyhow!(
            "No release asset matching '{}' for this platform. \
             Check https://github.com/ggml-org/llama.cpp/releases", pattern
        ))?;

    let url = asset["browser_download_url"].as_str()
        .ok_or_else(|| anyhow!("missing download URL"))?;

    println!("  Downloading: {}", asset["name"].as_str().unwrap_or("?"));

    let client = reqwest::Client::new();
    let bytes = client.get(url)
        .header("User-Agent", USER_AGENT)
        .send().await?
        .bytes().await?;

    std::fs::create_dir_all(dest.parent().unwrap())?;

    let is_gz = asset["name"].as_str()
        .map(|n| n.ends_with(".tar.gz"))
        .unwrap_or(false);

    if is_gz {
        let decoder = GzDecoder::new(bytes.as_ref());
        let mut archive = tar::Archive::new(decoder);

        let mut found = false;
        for entry in archive.entries()? {
            let mut entry = entry?;
            let name = entry.path().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();

            if name.ends_with(binary_name()) || name == binary_name() {
                entry.unpack(&dest)?;
                found = true;
            }

            #[cfg(target_os = "linux")]
            if name.ends_with(".so") || name.contains(".so.") {
                let lib_dir = crate::config::data_dir().join("lib");
                std::fs::create_dir_all(&lib_dir).ok();
                let lib_name = std::path::Path::new(&name)
                    .file_name().unwrap_or_default();
                let lib_dest = lib_dir.join(&lib_name);
                let entry_type = entry.header().entry_type();
                if entry_type.is_symlink() {
                    if let Ok(Some(link_target)) = entry.link_name() {
                        let _ = std::fs::remove_file(&lib_dest);
                        let _ = std::os::unix::fs::symlink(&link_target, &lib_dest);
                    }
                } else if entry_type.is_file() {
                    if let Ok(Some(link_target)) = entry.link_name() {
                        let _ = std::fs::remove_file(&lib_dest);
                        let _ = std::os::unix::fs::symlink(&link_target, &lib_dest);
                    } else {
                        let mut out = std::fs::File::create(&lib_dest)?;
                        std::io::copy(&mut entry, &mut out)?;
                    }
                }
            }
        }

        if !found {
            bail!("rpc-server binary not found inside the downloaded tarball");
        }
    } else {
        let cursor = std::io::Cursor::new(bytes.as_ref());
        let mut archive = zip::ZipArchive::new(cursor)?;

        let mut found = false;
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name().to_string();

            if name.ends_with(binary_name()) {
                let mut out = std::fs::File::create(&dest)?;
                std::io::copy(&mut file, &mut out)?;
                found = true;
            }

            #[cfg(target_os = "linux")]
            if name.ends_with(".so") || name.contains(".so.") {
                let lib_dir = crate::config::data_dir().join("lib");
                std::fs::create_dir_all(&lib_dir)?;
                let lib_name = std::path::Path::new(&name)
                    .file_name().unwrap_or_default();
                let lib_dest = lib_dir.join(lib_name);
                let mut out = std::fs::File::create(lib_dest)?;
                std::io::copy(&mut file, &mut out)?;
            }
        }

        if !found {
            bail!("rpc-server binary not found inside the downloaded zip");
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
    }

    if let Ok(mut cfg) = crate::config::load_config() {
        cfg.rpc_version = tag.clone();
        cfg.rpc_binary  = dest.to_string_lossy().to_string();
        let _ = crate::config::save_config(&cfg);
    }

    println!("rpc-server installed ({})", tag);
    Ok(())
}

pub fn spawn_rpc_server(binary: &Path, port: u16) -> Result<std::process::Child> {
    let mut cmd = std::process::Command::new(binary);
    cmd.arg("--host").arg("0.0.0.0")
       .arg("--port").arg(port.to_string());

    #[cfg(target_os = "linux")]
    {
        let lib_dir = crate::config::data_dir().join("lib");
        if lib_dir.exists() {
            cmd.env("LD_LIBRARY_PATH", lib_dir);
        }
    }

    Ok(cmd.spawn()?)
}