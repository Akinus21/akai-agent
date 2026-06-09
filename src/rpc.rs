use anyhow::{anyhow, bail, Result};
use std::path::{Path, PathBuf};
use flate2::read::GzDecoder;

const GITHUB_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
const USER_AGENT: &str = concat!("akai-agent/", env!("CARGO_PKG_VERSION"));

pub fn binary_name() -> &'static str {
    if cfg!(windows) { "rpc-server.exe" } else { "rpc-server" }
}

pub fn llama_server_name() -> &'static str {
    if cfg!(windows) { "llama-server.exe" } else { "llama-server" }
}

pub fn rpc_binary_path() -> PathBuf {
    crate::config::data_dir().join(binary_name())
}

pub fn llama_server_path() -> PathBuf {
    crate::config::data_dir().join(llama_server_name())
}

fn asset_pattern() -> &'static str {
    match (std::env::consts::OS, std::env:: consts::ARCH) {
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

async fn fetch_json(url: &str) -> Result<serde_json::Value> {
    let client = reqwest::Client::new();
    let resp = client.get(url)
        .header("User-Agent", USER_AGENT)
        .send().await?;
    if !resp.status().is_success() {
        bail!("GitHub API returned {} for {}", resp.status(), url);
    }
    Ok(resp.json().await?)
}

async fn fetch_latest_release() -> Result<serde_json::Value> {
    fetch_json(GITHUB_API).await
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

pub fn rpc_commit_hash() -> String {
    let path = rpc_binary_path();
    if !path.exists() {
        return String::new();
    }
    match std::process::Command::new(&path)
        .arg("--version")
        .output()
    {
        Ok(output) => {
            let ver_output = String::from_utf8_lossy(&output.stderr);
            for part in ver_output.split(|c: char| !c.is_ascii_hexdigit()) {
                if part.len() >= 7 {
                    return part.to_string();
                }
            }
            String::new()
        }
        Err(_) => String::new(),
    }
}

pub async fn ensure_rpc_server() -> Result<PathBuf> {
    let path = rpc_binary_path();
    let lib_dir = crate::config::data_dir().join("lib");

    if path.exists() && lib_dir.exists() && !has_missing_libs(&path) {
        return Ok(path);
    }

    // Always build from source on Linux to guarantee GPU compatibility
    #[cfg(target_os = "linux")]
    {
        println!("Building rpc-server from source (ensures GPU compatibility)...");
        if lib_dir.exists() {
            std::fs::remove_dir_all(&lib_dir).ok();
        }
        std::fs::remove_file(&path).ok();
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::create_dir_all(&lib_dir)?;

        match crate::build::build_from_source() {
            Ok(p) => return Ok(p),
            Err(e) => eprintln!("Source build failed: {}. Trying pre-built download.", e),
        }
    }

    if !path.exists() || has_missing_libs(&path) {
        std::fs::create_dir_all(path.parent().unwrap())?;
        download_latest().await?;
    }

    Ok(path)
}

fn has_missing_libs(binary: &Path) -> bool {
    use std::process::Command;
    let lib_dir = crate::config::data_dir().join("lib");
    let mut ld_path = lib_dir.to_string_lossy().to_string();
    for dir in &[
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/local/lib",
    ] {
        if std::path::Path::new(dir).exists() {
            ld_path.push_str(&format!(":{}", dir));
        }
    }
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        ld_path.push_str(&format!(":{}", existing));
    }
    if let Ok(output) = Command::new("ldd").arg(binary).env("LD_LIBRARY_PATH", &ld_path).output() {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if line.trim().contains("not found") {
                return true;
            }
        }
    }
    false
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
        let mut pending_symlinks: Vec<(std::path::PathBuf, String)> = Vec::new();

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
                    .file_name().unwrap_or_default().to_string_lossy().to_string();
                let lib_dest = lib_dir.join(&lib_name);
                let entry_type = entry.header().entry_type();
                if entry_type.is_symlink() {
                    // Store symlinks for later creation
                    if let Ok(Some(link_target)) = entry.link_name() {
                        let link_target_str = link_target.to_string_lossy().to_string();
                        pending_symlinks.push((lib_dest, link_target_str));
                    }
                } else if entry_type.is_file() {
                    let mut out = std::fs::File::create(&lib_dest)?;
                    std::io::copy(&mut entry, &mut out)?;
                    println!("Extracted shared lib: {}", lib_name);
                }
            }
        }

        // Create symlinks after all files are extracted
        for (dest, target) in &pending_symlinks {
            let _ = std::fs::remove_file(dest);
            if let Err(e) = std::os::unix::fs::symlink(target, dest) {
                eprintln!("Warning: failed to create symlink: {}", e);
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
    let binary = binary.to_path_buf();
    let lib_dir = crate::config::data_dir().join("lib");

    let mut cmd = std::process::Command::new(&binary);
    cmd.arg("--host").arg("0.0.0.0")
       .arg("--port").arg(port.to_string());

    #[cfg(target_os = "linux")]
    {
        let mut ld_path = lib_dir.to_string_lossy().to_string();
        for dir in &[
            "/home/linuxbrew/.linuxbrew/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
            "/lib/x86_64-linux-gnu",
            "/lib64",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu/dri",
            "/usr/lib/x86_64-linux-gnu/vulkan",
        ] {
            if std::path::Path::new(dir).exists() {
                ld_path.push_str(&format!(":{}", dir));
            }
        }
        if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
            ld_path.push_str(&format!(":{}", existing));
        }
        eprintln!("  LD_LIBRARY_PATH={}", ld_path);
        cmd.env("LD_LIBRARY_PATH", ld_path);
    }

    Ok(cmd.spawn()?)
}

pub async fn ensure_llama_server() -> Result<PathBuf> {
    let path = llama_server_path();

    if path.exists() && !has_missing_libs(&path) {
        return Ok(path);
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(built) = crate::build::build_from_source() {
            let _ = built;
            if path.exists() && !has_missing_libs(&path) {
                println!("llama-server available from source build");
                return Ok(path);
            }
            for search_dir in &[
                crate::build::source_dir().join("build").join("bin"),
                crate::build::source_dir().join("build"),
            ] {
                let llama_src = search_dir.join("llama-server");
                if llama_src.exists() {
                    std::fs::copy(&llama_src, &path)?;
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
                    }
                    println!("llama-server copied from source build");
                    return Ok(path);
                }
            }
            eprintln!("llama-server not found in source build, downloading pre-built...");
        } else {
            eprintln!("Source build failed, downloading pre-built llama-server...");
        }
    }

    println!("Downloading llama-server...");

    let release = fetch_latest_release().await?;
    let tag = release["tag_name"].as_str().unwrap_or("latest");
    let assets = release["assets"].as_array()
        .ok_or_else(|| anyhow!("no assets in release"))?;

    let pattern = asset_pattern();
    let asset = assets.iter()
        .find(|a| {
            let name = a["name"].as_str().unwrap_or("");
            glob_match(pattern, name)
        })
        .ok_or_else(|| anyhow!("llama.cpp release asset not found for pattern: {}", pattern))?;

    let url = asset["browser_download_url"].as_str()
        .ok_or_else(|| anyhow!("missing download URL"))?;

    let client = reqwest::Client::new();
    let bytes = client.get(url)
        .header("User-Agent", USER_AGENT)
        .send().await?
        .bytes().await?;

    let dest = &path;
    std::fs::create_dir_all(dest.parent().unwrap())?;

    #[cfg(target_os = "windows")]
    {
        let mut archive = zip::ZipArchive::new(bytes.as_ref())?;
        let mut found = false;
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name().to_string();
            if name.ends_with(llama_server_name()) {
                let mut out = std::fs::File::create(dest)?;
                std::io::copy(&mut file, &mut out)?;
                found = true;
            }
        }
        if !found {
            bail!("llama-server binary not found in downloaded zip");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let decoder = GzDecoder::new(bytes.as_ref());
        let mut archive = tar::Archive::new(decoder);
        let mut found = false;
        let lib_dir = crate::config::data_dir().join("lib");
        std::fs::create_dir_all(&lib_dir)?;
        let mut pending_symlinks: Vec<(std::path::PathBuf, String)> = Vec::new();
        for entry in archive.entries()? {
            let mut entry = entry?;
            let name = entry.path().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
            if name.ends_with("llama-server") || name.ends_with("llama-cli") {
                let mut out = std::fs::File::create(dest)?;
                std::io::copy(&mut entry, &mut out)?;
                found = true;
            } else if name.ends_with(".so") || name.contains(".so.") {
                let lib_name = entry.path().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).unwrap_or_default();
                if !lib_name.is_empty() {
                    let lib_dest = lib_dir.join(&lib_name);
                    let entry_type = entry.header().entry_type();
                    if entry_type.is_symlink() {
                        if let Ok(Some(link_target)) = entry.link_name() {
                            pending_symlinks.push((lib_dest, link_target.to_string_lossy().to_string()));
                        }
                    } else if entry_type.is_file() {
                        let mut out = std::fs::File::create(&lib_dest)?;
                        std::io::copy(&mut entry, &mut out)?;
                    }
                }
            }
        }

        // Create symlinks after all files are extracted
        for (dest, target) in &pending_symlinks {
            let _ = std::fs::remove_file(dest);
            if let Err(e) = std::os::unix::fs::symlink(target, dest) {
                eprintln!("Warning: failed to create symlink: {}", e);
            }
        }

        if !found {
            bail!("llama-server binary not found in downloaded tarball");
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))?;
    }

    println!("llama-server installed ({})", tag);
    Ok(dest.clone())
}

pub fn llama_server_port_flag(binary: &Path) -> &'static str {
    if let Ok(output) = std::process::Command::new(binary)
        .arg("--help")
        .output()
    {
        let help = String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
        if help.contains("--port") {
            return "--port";
        }
        if help.contains("\n-p ") {
            return "-p";
        }
    }
    "--port"
}

pub fn spawn_llama_server(binary: &Path, model_path: &str, n_gpu_layers: i32, port: u16, layer_offset: usize, num_layers: usize) -> Result<std::process::Child> {
    let port_flag = llama_server_port_flag(binary);
    let mut cmd = std::process::Command::new(binary);
    cmd.arg("-m").arg(model_path)
       .arg("-c").arg("4096")
       .arg("-ngl").arg(n_gpu_layers.to_string())
       .arg(port_flag).arg(port.to_string())
       .arg("--host").arg("0.0.0.0");
    
    if layer_offset > 0 {
        cmd.arg("--offset-layers").arg(layer_offset.to_string());
    }
    if num_layers > 0 {
        cmd.arg("--n-layers").arg(num_layers.to_string());
    }

    #[cfg(target_os = "linux")]
    {
        let lib_dir = crate::config::data_dir().join("lib");
        let mut ld_path = lib_dir.to_string_lossy().to_string();
        for dir in &[
            "/home/linuxbrew/.linuxbrew/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
            "/lib/x86_64-linux-gnu",
            "/lib64",
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu/dri",
            "/usr/lib/x86_64-linux-gnu/vulkan",
        ] {
            if std::path::Path::new(dir).exists() {
                ld_path.push_str(&format!(":{}", dir));
            }
        }
        if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
            ld_path.push_str(&format!(":{}", existing));
        }
        cmd.env("LD_LIBRARY_PATH", ld_path);
    }

    Ok(cmd.spawn()?)
}