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

fn is_ostree() -> bool {
    std::path::Path::new("/ostree").exists()
        || Command::new("which")
            .arg("rpm-ostree")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

fn is_container() -> bool {
    std::path::Path::new("/run/.containerenv").exists()
        || std::path::Path::new("/.dockerenv").exists()
        || std::fs::read_to_string("/proc/1/cgroup")
            .map(|c| c.contains("docker") || c.contains("lxc") || c.contains("distrobox"))
            .unwrap_or(false)
}

fn can_sudo() -> bool {
    Command::new("sudo")
        .args(["-n", "true"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn detect_pkg_manager() -> &'static str {
    if is_ostree() && !is_container() {
        return "rpm-ostree";
    }
    for cmd in &["apt-get", "dnf", "yum", "zypper", "pacman"] {
        if Command::new("which").arg(cmd).output().map(|o| o.status.success()).unwrap_or(false) {
            return cmd;
        }
    }
    "unknown"
}

fn detect_distro() -> String {
    for p in &["/etc/os-release", "/usr/lib/os-release"] {
        if let Ok(contents) = std::fs::read_to_string(p) {
            for line in contents.lines() {
                if line.starts_with("ID=") {
                    return line.trim_start_matches("ID=").trim('"').to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn detect_distro_version() -> String {
    for p in &["/etc/os-release", "/usr/lib/os-release"] {
        if let Ok(contents) = std::fs::read_to_string(p) {
            for line in contents.lines() {
                if line.starts_with("VERSION_ID=") {
                    return line.trim_start_matches("VERSION_ID=").trim('"').to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn nvidia_cuda_repo_url() -> String {
    let distro = detect_distro();
    let version = detect_distro_version();
    let (repo_distro, repo_ver) = match distro.as_str() {
        "fedora" => ("fedora", version.clone()),
        "rhel" | "centos" | "rocky" | "almalinux" => ("rhel", {
            let major = version.split('.').next().unwrap_or("9");
            format!("{}", major)
        }),
        "ubuntu" | "pop" => ("ubuntu", {
            let v: f32 = version.parse().unwrap_or(24.04);
            format!("{:.0}", v * 10)
        }),
        "debian" => ("debian", version.clone()),
        _ => ("rhel", "9".to_string()),
    };
    format!("https://developer.download.nvidia.com/compute/cuda/repos/{}{}/x86_64", repo_distro, repo_ver)
}

fn homebrew_install_build_tools() -> Result<()> {
    println!("  Installing build tools via Homebrew (works on atomic/immutable distros)...");
    let brew = find_in_paths("brew")
        .context("Homebrew not found. On atomic distros like Silverblue, install Homebrew:\n  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")?;

    let status = Command::new(&brew)
        .args(["install", "cmake", "gcc", "git"])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to run brew install")?;

    if !status.success() {
        bail!("Homebrew build tools installation failed");
    }
    Ok(())
}

pub fn install_cuda_toolkit() -> Result<()> {
    if has_nvcc() {
        println!("  CUDA toolkit already installed (nvcc found)");
        return Ok(());
    }

    println!("  Installing CUDA toolkit...");

    if is_ostree() && !is_container() {
        if !can_sudo() {
            bail!(
                "CUDA toolkit requires sudo on Silverblue/atomic distros.\n\
                 Options:\n\
                 1. Run inside a Distrobox/toolbox container (recommended)\n\
                 2. Run with sudo: rpm-ostree install cuda-toolkit (requires reboot)\n\
                 3. Install manually from https://developer.nvidia.com/cuda-downloads"
            );
        }

        println!("  Silverblue/atomic distro detected. CUDA requires layered packages via rpm-ostree.");
        println!("  This will require a REBOOT before the CUDA toolkit is available.");
        println!("  Alternatively, run akai-agent inside a Distrobox container for a seamless experience.");

        let status = Command::new("sudo")
            .args(["rpm-ostree", "install", "-y", "cuda-toolkit"])
            .status()
            .context("Failed to run rpm-ostree install")?;

        if status.success() {
            bail!(
                "CUDA toolkit installed via rpm-ostree. A REBOOT is required.\n\
                 After reboot, run: akai-agent start"
            );
        }
        bail!("rpm-ostree install failed. Try running inside a Distrobox container or install manually.");
    }

    let pkg_mgr = detect_pkg_manager();
    let status = match pkg_mgr {
        "apt-get" => {
            println!("  Adding NVIDIA CUDA repository...");
            Command::new("sudo")
                .args(["sh", "-c", "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && apt-get update -qq && apt-get install -y cuda-toolkit"])
                .status()
        }
        "dnf" => {
            let distro = detect_distro();
            let (repo_id, repo_url) = if distro == "fedora" {
                ("cuda-fedora", nvidia_cuda_repo_url())
            } else {
                ("cuda-rhel", nvidia_cuda_repo_url())
            };
            println!("  Adding NVIDIA CUDA repository...");
            Command::new("sudo")
                .args(["sh", "-c", &format!(
                    "dnf config-manager --add-repo={url} && dnf install -y cuda-toolkit",
                    url = repo_url
                )])
                .status()
        }
        "yum" => {
            Command::new("sudo")
                .args(["sh", "-c", &format!(
                    "yum-config-manager --add-repo={url} && yum install -y cuda-toolkit",
                    url = nvidia_cuda_repo_url()
                )])
                .status()
        }
        "zypper" => {
            Command::new("sudo")
                .args(["sh", "-c", "zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/ && zypper install -y cuda-toolkit"])
                .status()
        }
        "pacman" => {
            Command::new("sudo")
                .args(["pacman", "-S", "--noconfirm", "cuda"])
                .status()
        }
        _ => {
            bail!("Unsupported package manager. Install CUDA toolkit manually from https://developer.nvidia.com/cuda-downloads");
        }
    }.context("Failed to run package manager")?;

    if !status.success() {
        bail!("CUDA toolkit installation failed. Install manually from https://developer.nvidia.com/cuda-downloads");
    }

    if has_nvcc() {
        println!("  CUDA toolkit installed successfully");
        Ok(())
    } else {
        bail!("CUDA toolkit installed but nvcc not found in PATH.\n  Add /usr/local/cuda/bin to your PATH or install manually.");
    }
}

pub fn install_build_tools() -> Result<()> {
    if has_build_tools() {
        return Ok(());
    }

    println!("  Installing build tools (cmake, gcc, git)...");

    if is_ostree() && !is_container() {
        if find_in_paths("brew").is_some() {
            return homebrew_install_build_tools();
        }
        if can_sudo() {
            eprintln!("  On Silverblue/atomic distros, Homebrew is recommended for build tools (no reboot).");
            eprintln!("  Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"");
            eprintln!("  Attempting rpm-ostree install (will require reboot)...");
            let status = Command::new("sudo")
                .args(["rpm-ostree", "install", "-y", "cmake", "gcc", "gcc-c++", "git"])
                .status()
                .context("Failed to run rpm-ostree install")?;
            if status.success() {
                bail!("Build tools installed via rpm-ostree. A REBOOT is required.\n  After reboot, run: akai-agent start");
            }
        }
        bail!(
            "No Homebrew and no sudo on atomic distro.\n\
             Install Homebrew for a reboot-free experience:\n\
             /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\n\
             Or run inside a Distrobox/toolbox container."
        );
    }

    let pkg_mgr = detect_pkg_manager();
    let status = match pkg_mgr {
        "apt-get" => {
            Command::new("sudo")
                .args(["sh", "-c", "apt-get update -qq && apt-get install -y cmake build-essential git"])
                .status()
        }
        "dnf" | "yum" => {
            Command::new("sudo")
                .args(["sh", "-c", "dnf install -y cmake gcc gcc-c++ make git || yum install -y cmake gcc gcc-c++ make git"])
                .status()
        }
        "zypper" => {
            Command::new("sudo")
                .args(["sh", "-c", "zypper install -y cmake gcc gcc-c++ make git"])
                .status()
        }
        "pacman" => {
            Command::new("sudo")
                .args(["pacman", "-S", "--noconfirm", "cmake", "gcc", "make", "git"])
                .status()
        }
        _ => {
            bail!("Unsupported package manager. Install cmake, gcc, and git manually.");
        }
    }.context("Failed to run package manager")?;

    if !status.success() {
        bail!("Build tools installation failed. Install cmake, gcc, and git manually.");
    }

    if !has_build_tools() {
        bail!("Build tools installed but not found in PATH.\n  You may need to open a new terminal or add Homebrew to your PATH.");
    }

    println!("  Build tools installed successfully");
    Ok(())
}

pub fn build_from_source() -> Result<PathBuf> {
    let src = source_dir();
    let bin = crate::rpc::rpc_binary_path();
    let lib_dir = data_dir().join("lib");
    let env_path = path_with_homebrew();

    install_build_tools()?;

    let cuda_available = has_cuda();
    let nvcc_available = if cuda_available {
        if !has_nvcc() {
            match install_cuda_toolkit() {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("  CUDA toolkit installation failed: {}", e);
                    eprintln!("  Building without CUDA (CPU-only). GPU will not be used.");
                    false
                }
            }
        } else {
            true
        }
    } else {
        false
    };

    println!("Building rpc-server from source (CUDA: {})", nvcc_available);

    let git = find_in_paths("git").context("git not found")?;
    let cmake = find_in_paths("cmake").context("cmake not found")?;

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
    let _ = std::fs::remove_dir_all(&build);
    std::fs::create_dir_all(&build)?;

    println!("  Configuring build...");
    let mut cmake_cmd = Command::new(&cmake);
    cmake_cmd.arg("-B").arg(&build);
    cmake_cmd.arg("-S").arg(&src);
    cmake_cmd.arg("-DCMAKE_BUILD_TYPE=Release");
    cmake_cmd.arg("-DGGML_RPC=ON");
    if nvcc_available {
        cmake_cmd.arg("-DGGML_CUDA=ON");
    }
    cmake_cmd.env("PATH", &env_path);
    if nvcc_available {
        let ld_path = format!(
            "/usr/local/cuda/lib64:/home/linuxbrew/.linuxbrew/lib:{}",
            std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
        );
        cmake_cmd.env("LD_LIBRARY_PATH", &ld_path);
    }
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
        cfg.rpc_version = if nvcc_available { "source-cuda" } else { "source-cpu" }.to_string();
        cfg.rpc_binary = bin.to_string_lossy().to_string();
        let _ = crate::config::save_config(&cfg);
    }

    println!("rpc-server built from source (CUDA: {})", nvcc_available);
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