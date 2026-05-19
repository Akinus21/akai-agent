use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::process::Command;

const LLAMA_CPP_REPO: &str = "https://github.com/ggml-org/llama.cpp.git";

const HOMEBREW_PATHS: &[&str] = &[
    "/home/linuxbrew/.linuxbrew/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
];

fn data_dir() -> PathBuf {
    crate::config::data_dir()
}

fn source_dir() -> PathBuf {
    data_dir().join("llama.cpp")
}

fn build_dir() -> PathBuf {
    source_dir().join("build")
}

fn find_in_paths(cmd: &str) -> Option<PathBuf> {
    if let Ok(output) = Command::new("which").arg(cmd).output() {
        if output.status.success() {
            let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
    }
    for dir in HOMEBREW_PATHS {
        let p = PathBuf::from(dir).join(cmd);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

pub fn has_build_tools() -> bool {
    find_in_paths("cmake").is_some() && find_in_paths("cc").is_some() && find_in_paths("git").is_some()
}

pub fn has_cuda() -> bool {
    find_in_paths("nvidia-smi").is_some()
}

pub fn has_nvcc() -> bool {
    find_in_paths("nvcc").is_some()
}

pub fn needs_source_build() -> bool {
    if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        has_cuda()
    } else {
        false
    }
}

fn path_with_homebrew() -> String {
    let mut paths: Vec<String> = HOMEBREW_PATHS.iter().map(|s| s.to_string()).collect();
    if let Ok(default_path) = std::env::var("PATH") {
        paths.push(default_path);
    }
    paths.join(":")
}

pub fn build_from_source() -> Result<PathBuf> {
    let src = source_dir();
    let bin = crate::rpc::rpc_binary_path();
    let lib_dir = data_dir().join("lib");
    let env_path = path_with_homebrew();

    if !has_build_tools() {
        bail!(
            "Build tools not found.\n  \
             Install: brew install cmake gcc git\n  \
             For CUDA: install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
        );
    }

    let cuda_available = has_cuda();
    let nvcc_available = has_nvcc();

    if cuda_available && !nvcc_available {
        eprintln!("WARNING: NVIDIA GPU detected but CUDA toolkit (nvcc) not found.");
        eprintln!("  Install CUDA toolkit for GPU acceleration:");
        eprintln!("  https://developer.nvidia.com/cuda-downloads");
        eprintln!("  Building without CUDA (CPU-only)...");
    }

    println!("Building rpc-server from source (CUDA: {})", cuda_available && nvcc_available);

    let git = find_in_paths("git").unwrap();
    let cmake = find_in_paths("cmake").unwrap();

    if !src.exists() {
        println!("  Cloning llama.cpp repository...");
        let status = Command::new(&git)
            .args(["clone", "--depth", "1", LLAMA_CPP_REPO, &src.to_string_lossy()])
            .env("PATH", &env_path)
            .status()
            .context("Failed to run git clone")?;
        if !status.success() {
            bail!("git clone failed");
        }
    } else {
        println!("  Updating llama.cpp repository...");
        let _ = Command::new(&git)
            .args(["-C", &src.to_string_lossy(), "pull", "--ff-only"])
            .env("PATH", &env_path)
            .status();
    }

    let build = build_dir();
    std::fs::create_dir_all(&build)?;

    println!("  Configuring build...");
    let mut cmake_cmd = Command::new(&cmake);
    cmake_cmd.arg("-B").arg(&build);
    cmake_cmd.arg("-S").arg(&src);
    cmake_cmd.arg("-DCMAKE_BUILD_TYPE=Release");
    cmake_cmd.arg("-DGGML_RPC=ON");
    if cuda_available && nvcc_available {
        cmake_cmd.arg("-DGGML_CUDA=ON");
    }
    cmake_cmd.env("PATH", &env_path);
    let status = cmake_cmd.status().context("Failed to run cmake")?;
    if !status.success() {
        bail!("cmake configuration failed. Ensure build dependencies are installed.");
    }

    println!("  Building rpc-server (this may take a few minutes)...");
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());
    let status = Command::new(&cmake)
        .args(["--build", &build.to_string_lossy(), "--config", "Release", "-j", &nproc])
        .env("PATH", &env_path)
        .status()
        .context("Failed to run cmake --build")?;
    if !status.success() {
        bail!("Build failed");
    }

    let bin_dir = build.join("bin");
    let built_bin = find_binary(&bin_dir)
        .or_else(|_| find_binary(&build))
        .context("Built binary not found after compilation")?;

    println!("  Copying binary to {}", bin.display());
    std::fs::copy(&built_bin, &bin)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755))?;
    }

    std::fs::create_dir_all(&lib_dir)?;
    println!("  Copying shared libraries...");
    copy_libs(&build, &lib_dir)?;

    if let Ok(mut cfg) = crate::config::load_config() {
        cfg.rpc_version = if cuda_available && nvcc_available { "source-cuda" } else { "source-cpu" }.to_string();
        cfg.rpc_binary = bin.to_string_lossy().to_string();
        let _ = crate::config::save_config(&cfg);
    }

    println!("rpc-server built from source (CUDA: {})", cuda_available && nvcc_available);
    Ok(bin)
}

fn find_binary(dir: &std::path::Path) -> Result<PathBuf> {
    let names = ["rpc-server", "llama-rpc-server"];
    for name in names {
        let path = dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    bail!("Binary not found in {}", dir.display())
}

fn copy_libs(build_dir: &std::path::Path, lib_dir: &std::path::Path) -> Result<()> {
    for dir in &["bin", "."] {
        let search = build_dir.join(dir);
        if !search.exists() {
            continue;
        }
        copy_libs_from_dir(&search, lib_dir)?;
    }
    Ok(())
}

fn copy_libs_from_dir(search_dir: &std::path::Path, lib_dir: &std::path::Path) -> Result<()> {
    if let Ok(entries) = std::fs::read_dir(search_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with("libggml") && (name.ends_with(".so") || name.contains(".so.")) {
                let dest = lib_dir.join(path.file_name().unwrap());
                std::fs::copy(&path, &dest)?;
            }
        }
    }
    Ok(())
}