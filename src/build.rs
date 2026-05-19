use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::process::Command;

const LLAMA_CPP_REPO: &str = "https://github.com/ggml-org/llama.cpp.git";

fn data_dir() -> PathBuf {
    crate::config::data_dir()
}

fn source_dir() -> PathBuf {
    data_dir().join("llama.cpp")
}

fn build_dir() -> PathBuf {
    source_dir().join("build")
}

pub fn has_build_tools() -> bool {
    Command::new("cmake").arg("--version").output().is_ok()
        && Command::new("cc").arg("--version").output().is_ok()
}

pub fn has_cuda() -> bool {
    Command::new("nvidia-smi").output().is_ok()
}

pub fn needs_source_build() -> bool {
    if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        has_cuda()
    } else {
        false
    }
}

pub fn build_from_source() -> Result<PathBuf> {
    let src = source_dir();
    let bin = crate::rpc::rpc_binary_path();
    let lib_dir = data_dir().join("lib");

    if !has_build_tools() {
        bail!(
            "Build tools not found. Install cmake and a C compiler:\n  \
             sudo apt install cmake build-essential\n  \
             For CUDA: install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
        );
    }

    let cuda_available = has_cuda();
    println!("Building rpc-server from source (CUDA: {})", cuda_available);

    if !src.exists() {
        println!("  Cloning llama.cpp repository...");
        let status = Command::new("git")
            .args(["clone", "--depth", "1", LLAMA_CPP_REPO, &src.to_string_lossy()])
            .status()
            .context("Failed to run git clone")?;
        if !status.success() {
            bail!("git clone failed");
        }
    } else {
        println!("  Updating llama.cpp repository...");
        let _ = Command::new("git")
            .args(["-C", &src.to_string_lossy(), "pull", "--ff-only"])
            .status();
    }

    let build = build_dir();
    std::fs::create_dir_all(&build)?;

    println!("  Configuring build...");
    let mut cmake = Command::new("cmake");
    cmake.arg("-B").arg(&build);
    cmake.arg("-S").arg(&src);
    cmake.arg("-DCMAKE_BUILD_TYPE=Release");
    cmake.arg("-DGGML_RPC=ON");
    if cuda_available {
        cmake.arg("-DGGML_CUDA=ON");
    }
    let status = cmake.status().context("Failed to run cmake")?;
    if !status.success() {
        bail!("cmake configuration failed. Ensure build dependencies are installed.");
    }

    println!("  Building rpc-server (this may take a few minutes)...");
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());
    let status = Command::new("cmake")
        .args(["--build", &build.to_string_lossy(), "--config", "Release", "-j", &nproc])
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
        cfg.rpc_version = "source-build".to_string();
        cfg.rpc_binary = bin.to_string_lossy().to_string();
        let _ = crate::config::save_config(&cfg);
    }

    println!("rpc-server built from source (CUDA: {})", cuda_available);
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