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

fn nvidia_gpu_compute_cap() -> &'static str {
    let output = match Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return "75",
    };
    let name = String::from_utf8_lossy(&output.stdout).to_lowercase();
    if name.contains("rtx 5090") || name.contains("rtx 5080") || name.contains("rtx 5070") {
        "120"
    } else if name.contains("rtx 4090") || name.contains("rtx 4080") || name.contains("rtx 4070") || name.contains("rtx 4060") {
        "89"
    } else if name.contains("rtx 3090") || name.contains("rtx 3080") || name.contains("rtx 3070") || name.contains("rtx 3060") {
        "86"
    } else if name.contains("rtx 2080") || name.contains("rtx 2070") || name.contains("rtx 2060") {
        "75"
    } else if name.contains("gtx 1080") || name.contains("gtx 1070") || name.contains("gtx 1060") {
        "61"
    } else if name.contains("a100") || name.contains("a10") || name.contains("a30") {
        "80"
    } else if name.contains("h100") || name.contains("h200") {
        "90"
    } else if name.contains("l40") || name.contains("l4") {
        "89"
    } else {
        "75"
    }
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

fn sudo_user() -> Option<String> {
    std::env::var("SUDO_USER").ok().filter(|u| !u.is_empty())
}

fn sudo_user_home() -> Option<PathBuf> {
    sudo_user().and_then(|u| {
        let output = Command::new("getent")
            .args(["passwd", &u])
            .output()
            .ok()?;
        let line = String::from_utf8_lossy(&output.stdout);
        let home = line.split(':').nth(5)?;
        if home.is_empty() {
            None
        } else {
            Some(PathBuf::from(home))
        }
    })
}

fn effective_data_dir() -> PathBuf {
    if let Some(home) = sudo_user_home() {
        home.join(".local").join("share").join("akai-agent")
    } else {
        data_dir()
    }
}

fn effective_source_dir() -> PathBuf {
    effective_data_dir().join("src").join("llama.cpp")
}

fn run_distrobox(args: &[String]) -> Result<std::process::ExitStatus> {
    let path = path_with_homebrew();
    if let Some(user) = sudo_user() {
        let sudo_args: Vec<String> = vec![
            "-u".into(),
            user.clone(),
            "distrobox".into(),
        ];
        let mut cmd = Command::new("sudo");
        cmd.args(&sudo_args)
            .args(args)
            .env("PATH", &path);
        if let Some(home) = sudo_user_home() {
            cmd.env("HOME", home.to_string_lossy().to_string());
        }
        cmd.status().context("Failed to run distrobox command via sudo")
    } else {
        Command::new("distrobox")
            .args(args)
            .env("PATH", &path)
            .status()
            .context("Failed to run distrobox command")
    }
}

fn run_distrobox_output(args: &[String]) -> Result<std::process::Output> {
    let path = path_with_homebrew();
    if let Some(user) = sudo_user() {
        let sudo_args: Vec<String> = vec![
            "-u".into(),
            user.clone(),
            "distrobox".into(),
        ];
        let mut cmd = Command::new("sudo");
        cmd.args(&sudo_args)
            .args(args)
            .env("PATH", &path);
        if let Some(home) = sudo_user_home() {
            cmd.env("HOME", home.to_string_lossy().to_string());
        }
        cmd.output().context("Failed to run distrobox command via sudo")
    } else {
        Command::new("distrobox")
            .args(args)
            .env("PATH", &path)
            .output()
            .context("Failed to run distrobox command")
    }
}

fn ensure_akai_container() -> Result<String> {
    ensure_distrobox()?;
    let container_name = "akai";

    let list_args: Vec<String> = vec!["list".into(), "--no-header".into()];
    let output = run_distrobox_output(&list_args)?;
    let listing = String::from_utf8_lossy(&output.stdout);
    if listing.lines().any(|l| l.contains(container_name)) {
        println!("  Distrobox container '{}' already exists", container_name);
        return Ok(container_name.to_string());
    }

    println!("  Creating distrobox container '{}'...", container_name);
    let create_args: Vec<String> = vec!["create".into(), "--name".into(), container_name.into(), "--image".into(), "ubuntu:24.04".into(), "--yes".into()];
    let status = run_distrobox(&create_args)?;
    if !status.success() {
        bail!("Failed to create distrobox container '{}'. Try running without sudo or install distrobox first.", container_name);
    }

    println!("  Installing build tools in container...");
    let install_cmd = "sudo apt-get update -qq && sudo apt-get install -y cmake gcc g++ git wget curl";
    let enter_args: Vec<String> = vec!["enter".into(), container_name.into(), "--".into(), "sh".into(), "-c".into(), install_cmd.into()];
    let status = run_distrobox(&enter_args)?;
    if !status.success() {
        bail!("Failed to install build tools in container");
    }
    Ok(container_name.to_string())
}

pub fn build_in_distrobox() -> Result<PathBuf> {
    let container = ensure_akai_container()?;
    let bin = crate::rpc::rpc_binary_path();
    let eff_data = effective_data_dir();
    let eff_src = effective_source_dir();

    let driver_ver = nvidia_driver_version().unwrap_or_default();
    let (cuda_major, cuda_minor) = cuda_version_for_driver(&driver_ver);
    let cuda_pkg = format!("cuda-toolkit-{}-{}", cuda_major, cuda_minor);

    println!("  Detected NVIDIA driver {}, installing CUDA {}.{}", driver_ver, cuda_major, cuda_minor);

    println!("  Installing CUDA toolkit {} in container...", cuda_pkg);
    let cuda_install_cmd = format!(
        "sudo apt-get update -qq && \
         wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && \
         sudo dpkg -i /tmp/cuda-keyring.deb && \
         sudo apt-get update -qq && \
         sudo apt-get install -y {pkg} && \
         sudo ldconfig",
        pkg = cuda_pkg
    );
    let status = run_distrobox(&vec!["enter".into(), container.clone(), "--".into(), "sh".into(), "-c".into(), cuda_install_cmd])?;
    if !status.success() {
        println!("  CUDA toolkit install failed, trying individual packages...");
        let fallback = format!(
            "sudo apt-get update -qq && \
             wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && \
             sudo dpkg -i /tmp/cuda-keyring.deb && \
             sudo apt-get update -qq && \
             sudo apt-get install -y --fix-broken && \
             for pkg in cuda-nvcc-{maj}-{min} cuda-cudart-{maj}-{min} cuda-cudart-dev-{maj}-{min} \
                        cuda-cccl-{maj}-{min} cuda-cupti-{maj}-{min} \
                        libcublas-dev-{maj}-{min} libcublas-{maj}-{min}; do \
               sudo apt-get install -y $pkg 2>/dev/null || true; \
             done && \
             sudo ldconfig",
            maj = cuda_major, min = cuda_minor
        );
        let status = run_distrobox(&vec!["enter".into(), container.clone(), "--".into(), "sh".into(), "-c".into(), fallback])?;
        if !status.success() {
            eprintln!("  CUDA installation failed. Building CPU-only.");
        }
    }

    let has_nvcc_out = run_distrobox_output(&vec!["enter".into(), container.clone(), "--".into(), "sh".into(), "-c".into(), "ls /usr/local/cuda*/bin/nvcc 2>/dev/null || which nvcc 2>/dev/null || echo ''".into()])?;
    let has_nvcc = !String::from_utf8_lossy(&has_nvcc_out.stdout).trim().is_empty();

    let arch = nvidia_gpu_compute_cap();
    let cmake_args = if has_nvcc {
        format!("-DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES={}", arch)
    } else {
        "-DGGML_RPC=ON".to_string()
    };

    println!("  Building rpc-server in distrobox (CUDA: {})...", has_nvcc);

    let llama_src = format!("{}", eff_src.to_string_lossy());
    let data_path = format!("{}", eff_data.to_string_lossy());

    if let Some(user) = sudo_user() {
        let eff_path_str = format!("{}", eff_data.to_string_lossy());
        println!("  Fixing ownership of {} for user {}...", eff_path_str, user);
        let _ = Command::new("chown")
            .args(["-R", &format!("{}:", user), &eff_path_str])
            .status();
    }

    let mkdir_cmd = format!("mkdir -p '{data}/lib' '{src}'", data = data_path, src = llama_src);
    let status = run_distrobox(&vec!["enter".into(), container.clone(), "--".into(), "sh".into(), "-c".into(), mkdir_cmd])?;
    if !status.success() {
        bail!("Failed to create build directories in container");
    }

    let build_cmd = format!(
        "export PATH=/usr/local/cuda-{maj}.{min}/bin:$PATH && \
         if [ ! -d '{src}/.git' ]; then \
           git clone --depth 1 {repo} '{src}'; \
         fi && \
         mkdir -p '{src}/build' && \
         cd '{src}/build' && \
         for libdir in /run/host/usr/lib/x86_64-linux-gnu /run/host/usr/lib64 /run/host/lib/x86_64-linux-gnu /run/host/lib64 /usr/lib/x86_64-linux-gnu; do \
           if [ -f \"$libdir/libcuda.so.1\" ]; then \
             sudo ln -sf \"$libdir/libcuda.so.1\" /usr/local/lib/libcuda.so.1 2>/dev/null; \
             sudo ln -sf \"$libdir/libcuda.so.1\" /usr/local/lib/libcuda.so 2>/dev/null; \
             break; \
           fi; \
         done && \
         sudo ldconfig 2>/dev/null; \
         cmake .. -DCMAKE_BUILD_TYPE=Release {cmake_args} && \
         cmake --build . --config Release -j$(nproc) && \
         cp '{src}/build/bin/rpc-server' '{data}/rpc-server' 2>/dev/null || \
         cp '{src}/build/bin/llama-rpc-server' '{data}/rpc-server' 2>/dev/null || true && \
         mkdir -p '{data}/lib' && \
         for dir in '{src}/build/bin' '{src}/build'; do \
           for f in \"$dir\"/libggml*.so \"$dir\"/libggml*.so.*; do \
             [ -f \"$f\" ] && cp \"$f\" '{data}/lib/' 2>/dev/null || true; \
           done; \
         done",
        maj = cuda_major, min = cuda_minor,
        src = llama_src,
        repo = LLAMA_CPP_REPO,
        cmake_args = cmake_args,
        data = data_path
    );

    let status = run_distrobox(&vec!["enter".into(), container, "--".into(), "sh".into(), "-c".into(), build_cmd])?;
    if !status.success() {
        bail!("Build failed in distrobox");
    }

    let eff_bin = eff_data.join("rpc-server");
    if !eff_bin.exists() {
        bail!("Built binary not found at {}", eff_bin.display());
    }

    if eff_data != data_dir() {
        std::fs::create_dir_all(&data_dir())?;
        let target = data_dir().join("rpc-server");
        std::fs::copy(&eff_bin, &target)?;
        let eff_lib = eff_data.join("lib");
        let target_lib = data_dir().join("lib");
        std::fs::create_dir_all(&target_lib)?;
        if eff_lib.exists() {
            for entry in std::fs::read_dir(&eff_lib)? {
                let entry = entry?;
                if entry.file_name().to_string_lossy().contains(".so") {
                    std::fs::copy(entry.path(), target_lib.join(entry.file_name()))?;
                }
            }
        }
        std::fs::copy(&eff_bin, &bin)?;
    } else {
        std::fs::copy(&eff_bin, &bin)?;
    }
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
    if is_ostree() && !is_container() {
        println!("  Atomic/immutable distro detected. Using distrobox for build...");
        return build_in_distrobox();
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

        let cuda_ver = nvidia_driver_version()
            .map(|v| {
                let (maj, min) = cuda_version_for_driver(&v);
                format!("{}.{}", maj, min)
            })
            .unwrap_or_else(|| "12.4".to_string());
        let cuda_root = format!("/usr/local/cuda-{}", cuda_ver);
        let ld_path = format!(
            "{}/lib64:/usr/local/cuda/lib64:/home/linuxbrew/.linuxbrew/lib:{}",
            cuda_root,
            std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
        );
        cmake_cmd.env("LD_LIBRARY_PATH", &ld_path);
        cmake_cmd.arg(format!("-DCUDAToolkit_ROOT={}", cuda_root));
        cmake_cmd.arg(format!("-DCMAKE_CUDA_COMPILER={}/bin/nvcc", cuda_root));
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