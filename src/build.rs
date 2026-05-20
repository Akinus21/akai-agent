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
                    return line.trim_start_matches("ID=").trim_matches('"').to_string();
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
                    return line.trim_start_matches("VERSION_ID=").trim_matches('"').to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn nvidia_driver_version() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if version.is_empty() || version.contains("N/A") {
        return None;
    }
    Some(version)
}

fn cuda_version_for_driver(driver_version: &str) -> (u32, u32) {
    let major: u32 = driver_version.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(550);
    if major >= 580 {
        (13, 0)
    } else if major >= 570 {
        (12, 8)
    } else if major >= 565 {
        (12, 6)
    } else if major >= 555 {
        (12, 5)
    } else {
        (12, 4)
    }
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
        "ubuntu" | "pop" | "linuxmint" => ("ubuntu", {
            let v: f32 = version.parse().unwrap_or(24.04);
            format!("{:.0}", v * 10.0)
        }),
        "debian" => ("debian", version.clone()),
        _ => ("rhel", "9".to_string()),
    };
    format!("https://developer.download.nvidia.com/compute/cuda/repos/{}{}/x86_64", repo_distro, repo_ver)
}

fn has_distrobox() -> bool {
    find_in_paths("distrobox").is_some()
}

fn ensure_distrobox() -> Result<()> {
    if has_distrobox() {
        return Ok(());
    }
    println!("  Distrobox not found. Installing...");
    let url = "https://distrobox.it/prereqs/shell/distrobox-install";
    let status = Command::new("curl")
        .args(["-fsSL", url])
        .env("PATH", path_with_homebrew())
        .output()
        .context("Failed to download distrobox install script")?;
    if !status.status.success() {
        bail!("Failed to download distrobox");
    }
    let tmp_script = std::env::temp_dir().join("distrobox-install.sh");
    let status = Command::new("curl")
        .args(["-fsSL", "-o", &tmp_script.to_string_lossy(), url])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to download distrobox")?;
    if !status.success() {
        bail!("Failed to download distrobox install script");
    }
    let status = Command::new("sh")
        .arg(&tmp_script)
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to run distrobox install")?;
    if !status.success() {
        bail!("distrobox installation failed");
    }
    let _ = std::fs::remove_file(&tmp_script);
    println!("  Distrobox installed.");
    Ok(())
}

fn ensure_akai_container() -> Result<String> {
    ensure_distrobox()?;
    let container_name = "akai";
    let output = Command::new("distrobox")
        .args(["list", "--no-header"])
        .env("PATH", path_with_homebrew())
        .output()
        .context("Failed to list distrobox containers")?;
    let listing = String::from_utf8_lossy(&output.stdout);
    if listing.lines().any(|l| l.contains(container_name)) {
        println!("  Distrobox container '{}' already exists", container_name);
        return Ok(container_name.to_string());
    }
    println!("  Creating distrobox container '{}'...", container_name);
    let distro = if detect_distro() == "silverblue" || detect_distro() == "bluefin" {
        "ubuntu:24.04"
    } else {
        "ubuntu:24.04"
    };
    let status = Command::new("distrobox")
        .args(["create", "--name", container_name, "--image", distro, "--yes"])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to create distrobox container")?;
    if !status.success() {
        bail!("Failed to create distrobox container '{}'", container_name);
    }
    println!("  Installing build tools in container...");
    let status = Command::new("distrobox")
        .args(["enter", container_name, "--", "sh", "-c",
            "apt-get update -qq && apt-get install -y cmake gcc g++ git wget curl"])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to install build tools in container")?;
    if !status.success() {
        bail!("Failed to install build tools in container");
    }
    Ok(container_name.to_string())
}

pub fn build_in_distrobox() -> Result<PathBuf> {
    let container = ensure_akai_container()?;
    let bin = crate::rpc::rpc_binary_path();
    let lib_dir = data_dir().join("lib");
    let src = source_dir();
    let build = build_dir();
    let data = data_dir();

    let driver_ver = nvidia_driver_version().unwrap_or_default();
    let (cuda_major, cuda_minor) = cuda_version_for_driver(&driver_ver);
    let cuda_pkg = format!("cuda-toolkit-{}-{}", cuda_major, cuda_minor);

    println!("  Detected NVIDIA driver {}, installing CUDA {}.{}", driver_ver, cuda_major, cuda_minor);

    println!("  Installing CUDA toolkit {} in container...", cuda_pkg);
    let cuda_install_cmd = format!(
        "apt-get update -qq && \
         wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && \
         dpkg -i /tmp/cuda-keyring.deb && \
         apt-get update -qq && \
         apt-get install -y {pkg} && \
         ldconfig",
        pkg = cuda_pkg
    );
    let status = Command::new("distrobox")
        .args(["enter", &container, "--", "sh", "-c", &cuda_install_cmd])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to install CUDA toolkit in container")?;
    if !status.success() {
        println!("  CUDA toolkit install failed, trying individual packages...");
        let fallback = format!(
            "apt-get update -qq && \
             wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && \
             dpkg -i /tmp/cuda-keyring.deb && \
             apt-get update -qq && \
             apt-get install -y --fix-broken && \
             for pkg in cuda-nvcc-{maj}-{min} cuda-cudart-{maj}-{min} cuda-cudart-dev-{maj}-{min} \
                        cuda-cccl-{maj}-{min} cuda-cupti-{maj}-{min} \
                        libcublas-dev-{maj}-{min} libcublas-{maj}-{min}; do \
               apt-get install -y $pkg 2>/dev/null || true; \
             done && \
             ldconfig",
            maj = cuda_major, min = cuda_minor
        );
        let status = Command::new("distrobox")
            .args(["enter", &container, "--", "sh", "-c", &fallback])
            .env("PATH", path_with_homebrew())
            .status()
            .context("Failed to install CUDA packages in container")?;
        if !status.success() {
            eprintln!("  CUDA installation failed. Building CPU-only.");
        }
    }

    let mount_src = format!("--volume={}:/opt/llama.cpp", src.to_string_lossy());
    let mount_data = format!("--volume={}:/opt/akai-data", data.to_string_lossy());

    let has_nvcc_out = Command::new("distrobox")
        .args(["enter", &container, "--", "sh", "-c", "which nvcc 2>/dev/null || echo ''"])
        .env("PATH", path_with_homebrew())
        .output()
        .context("Failed to check for nvcc")?;
    let has_nvcc = !String::from_utf8_lossy(&has_nvcc_out.stdout).trim().is_empty();

    let cmake_args = if has_nvcc {
        "-DGGML_CUDA=ON -DGGML_RPC=ON"
    } else {
        "-DGGML_RPC=ON"
    };

    println!("  Building rpc-server in distrobox (CUDA: {})...", has_nvcc);

    let build_cmd = format!(
        "if [ ! -d /opt/llama.cpp/.git ]; then \
           git clone --depth 1 {repo} /opt/llama.cpp; \
         fi && \
         mkdir -p /opt/llama.cpp/build && \
         cd /opt/llama.cpp/build && \
         cmake .. -DCMAKE_BUILD_TYPE=Release {cmake_args} && \
         cmake --build . --config Release -j$(nproc)",
        repo = LLAMA_CPP_REPO,
        cmake_args = cmake_args
    );

    let status = Command::new("distrobox")
        .args(["enter", &container, "--", "sh", "-c", &build_cmd])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to build in distrobox")?;
    if !status.success() {
        bail!("Build failed in distrobox");
    }

    let container_src = "/opt/llama.cpp";
    let copy_cmd = format!(
        "cp {container_src}/build/bin/rpc-server {dest}/rpc-server 2>/dev/null || \
         cp {container_src}/build/bin/llama-rpc-server {dest}/rpc-server 2>/dev/null || true",
        container_src = container_src,
        dest = "/opt/akai-data"
    );
    Command::new("distrobox")
        .args(["enter", &container, "--", "sh", "-c", &copy_cmd])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to copy binary from container")?;

    let copy_libs = format!(
        "for dir in {container_src}/build/bin {container_src}/build; do \
           for f in \"$dir\"/libggml*.so \"$dir\"/libggml*.so.*; do \
             [ -f \"$f\" ] && cp \"$f\" /opt/akai-data/lib/ 2>/dev/null || true; \
           done; \
         done",
        container_src = container_src
    );
    std::fs::create_dir_all(&lib_dir)?;
    Command::new("distrobox")
        .args(["enter", &container, "--", "sh", "-c", &copy_libs])
        .env("PATH", path_with_homebrew())
        .status()
        .context("Failed to copy libs from container")?;

    let built_bin = data_dir().join("rpc-server");
    if !built_bin.exists() {
        bail!("Built binary not found in data dir");
    }
    std::fs::copy(&built_bin, &bin)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755))?;
    }

    let cuda_label = if has_nvcc { "source-cuda" } else { "source-cpu" };
    if let Ok(mut cfg) = crate::config::load_config() {
        cfg.rpc_version = cuda_label.to_string();
        cfg.rpc_binary = bin.to_string_lossy().to_string();
        let _ = crate::config::save_config(&cfg);
    }

    println!("  rpc-server built from source in distrobox (CUDA: {})", has_nvcc);
    Ok(bin)
}

fn homebrew_install_build_tools() -> Result<()> {
    println!("  Installing build tools via Homebrew...");
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
    let driver_ver = nvidia_driver_version().unwrap_or_default();
    let (cuda_major, cuda_minor) = cuda_version_for_driver(&driver_ver);
    println!("  Detected NVIDIA driver {}, selecting CUDA {}.{}", driver_ver, cuda_major, cuda_minor);

    if is_ostree() && !is_container() {
        if !can_sudo() {
            bail!(
                "CUDA toolkit requires sudo on Silverblue/atomic distros.\n\
                 Options:\n\
                 1. Run inside a Distrobox container (recommended)\n\
                 2. Run with sudo: rpm-ostree install cuda-toolkit (requires reboot)\n\
                 3. Install manually from https://developer.nvidia.com/cuda-downloads"
            );
        }
        println!("  Silverblue/atomic distro detected. Installing via distrobox...");
        return build_in_distrobox().map(|_| ());
    }

    let pkg_mgr = detect_pkg_manager();
    let cuda_pkg = format!("cuda-toolkit-{}-{}", cuda_major, cuda_minor);
    let status = match pkg_mgr {
        "apt-get" => {
            let repo_url = nvidia_cuda_repo_url();
            println!("  Adding NVIDIA CUDA repository...");
            Command::new("sudo")
                .args(["sh", "-c", &format!(
                    "wget -q {url}/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && \
                     dpkg -i /tmp/cuda-keyring.deb && \
                     apt-get update -qq && \
                     apt-get install -y {pkg}",
                    url = repo_url, pkg = cuda_pkg
                )])
                .status()
        }
        "rpm-ostree" => {
            Command::new("sudo")
                .args(["rpm-ostree", "install", "-y", &cuda_pkg])
                .status()
        }
        "dnf" => {
            let repo_url = nvidia_cuda_repo_url();
            Command::new("sudo")
                .args(["sh", "-c", &format!(
                    "dnf config-manager --add-repo={url} && dnf install -y {pkg}",
                    url = repo_url, pkg = cuda_pkg
                )])
                .status()
        }
        "yum" => {
            Command::new("sudo")
                .args(["sh", "-c", &format!(
                    "yum-config-manager --add-repo={url} && yum install -y {pkg}",
                    url = nvidia_cuda_repo_url(), pkg = cuda_pkg
                )])
                .status()
        }
        "zypper" => {
            Command::new("sudo")
                .args(["sh", "-c", &format!(
                    "zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/ && zypper install -y {pkg}",
                    pkg = cuda_pkg
                )])
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
    if (is_ostree() || is_container()) && !is_container() {
        if is_ostree() && !is_container() {
            println!("  Atomic/immutable distro detected. Using distrobox for build...");
            return build_in_distrobox();
        }
    }

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
            "/usr/local/cuda/lib64:/usr/local/cuda-{}.{}{/lib64}:/home/linuxbrew/.linuxbrew/lib:{}",
            nvidia_driver_version().and_then(|v| {
                let (maj, min) = cuda_version_for_driver(&v);
                Some(format!("{}.{}", maj, min))
            }).unwrap_or_else(|| "12.4".to_string()),
            "/home/linuxbrew/.linuxbrew/lib",
            std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
        );
        let cuda_root = format!("/usr/local/cuda-{}", 
            nvidia_driver_version().and_then(|v| {
                let (maj, min) = cuda_version_for_driver(&v);
                Some(format!("{}.{}", maj, min))
            }).unwrap_or_else(|| "12.4".to_string())
        );
        let ld_path = format!(
            "{cuda_root}/lib64:/usr/local/cuda/lib64:/home/linuxbrew/.linuxbrew/lib:{}",
            std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
        );
        cmake_cmd.env("LD_LIBRARY_PATH", &ld_path);
        let _ = cmake_cmd.arg(format!("-DCUDAToolkit_ROOT={}", cuda_root));
        let nvcc_path = format!("{}/bin/nvcc", cuda_root);
        let _ = cmake_cmd.arg(format!("-DCMAKE_CUDA_COMPILER={}", nvcc_path));
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