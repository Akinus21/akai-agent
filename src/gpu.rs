use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    Cuda,
    Vulkan,
    Metal,
    Cpu,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda  => write!(f, "cuda"),
            GpuBackend::Vulkan => write!(f, "vulkan"),
            GpuBackend::Metal  => write!(f, "metal"),
            GpuBackend::Cpu    => write!(f, "cpu"),
        }
    }
}

pub struct GpuInfo {
    pub has_gpu: bool,
    pub name: String,
    pub vram_gb: f64,
    pub backend: GpuBackend,
}

pub fn detect_gpu() -> GpuInfo {
    if let Some(info) = detect_nvidia() {
        return info;
    }
    if let Some(info) = detect_amd() {
        return info;
    }
    if let Some(info) = detect_intel() {
        return info;
    }
    if let Some(info) = detect_apple() {
        return info;
    }

    GpuInfo {
        has_gpu: false,
        name: "CPU only".to_string(),
        vram_gb: 0.0,
        backend: GpuBackend::Cpu,
    }
}

fn detect_nvidia() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim().split(',').collect::<Vec<_>>();

    if line.len() != 2 {
        return None;
    }

    let name = line[0].trim().to_string();
    let vram_str = line[1].trim().replace("MiB", "").replace("B", "");
    let vram_gb = vram_str.trim().parse::<f64>().ok()? / 1024.0;

    Some(GpuInfo {
        has_gpu: true,
        name,
        vram_gb,
        backend: GpuBackend::Cuda,
    })
}

fn detect_amd() -> Option<GpuInfo> {
    if let Some(info) = detect_amd_rocm() {
        return Some(info);
    }
    detect_amd_vulkan()
}

fn detect_amd_rocm() -> Option<GpuInfo> {
    let output = Command::new("rocm-smi")
        .args(["--query-gpu=name,memory.total", "--csv"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim().split('\n').nth(1)?.split(',').collect::<Vec<_>>();

    if line.len() < 2 {
        return None;
    }

    let name = line[0].trim().to_string();
    let vram_str = line[1].trim().replace("MiB", "").replace("B", "");
    let vram_gb = vram_str.trim().parse::<f64>().ok()? / 1024.0;

    Some(GpuInfo {
        has_gpu: true,
        name,
        vram_gb,
        backend: GpuBackend::Vulkan,
    })
}

fn detect_amd_vulkan() -> Option<GpuInfo> {
    for cmd in &["vulkaninfo", "amd-vulkan-info"] {
        if let Ok(output) = Command::new(cmd).arg("--summary").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let lower = line.to_lowercase();
                    if lower.contains("amd") || lower.contains("radeon") || lower.contains("navi") || lower.contains("renoir") || lower.contains("cedar") || lower.contains("topaz") || lower.contains("tonga") || lower.contains("fiji") || lower.contains("polaris") || lower.contains("vega") || lower.contains("gfx") {
                        if lower.contains("gpu") || lower.contains("device") || lower.contains("apu") || lower.contains("integrated") || lower.contains("discrete") {
                            let vram = detect_amd_vram_from_sysfs().unwrap_or(8.0);
                            return Some(GpuInfo {
                                has_gpu: true,
                                name: "AMD GPU (Vulkan)".to_string(),
                                vram_gb: vram,
                                backend: GpuBackend::Vulkan,
                            });
                        }
                    }
                }
            }
        }
    }

    if std::path::Path::new("/sys/class/drm/card0/device/vendor").exists() {
        if let Ok(vendor) = std::fs::read_to_string("/sys/class/drm/card0/device/vendor") {
            if vendor.trim() == "0x1002" || vendor.trim() == "0x1022" {
                let vram = detect_amd_vram_from_sysfs().unwrap_or(8.0);
                return Some(GpuInfo {
                    has_gpu: true,
                    name: "AMD GPU (Vulkan)".to_string(),
                    vram_gb: vram,
                    backend: GpuBackend::Vulkan,
                });
            }
        }
    }

    None
}

fn detect_amd_vram_from_sysfs() -> Option<f64> {
    for entry in std::fs::read_dir("/sys/class/drm").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains("-") {
            continue;
        }
        let vram_path = entry.path().join("device/mem_info_vram_total");
        if let Ok(vram_str) = std::fs::read_to_string(&vram_path) {
            if let Ok(vram_bytes) = vram_str.trim().parse::<u64>() {
                return Some(vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
            }
        }
    }

    let vram_path = std::path::Path::new("/sys/class/drm/card0/device/mem_info_vram_total");
    if let Ok(vram_str) = std::fs::read_to_string(vram_path) {
        if let Ok(vram_bytes) = vram_str.trim().parse::<u64>() {
            return Some(vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        }
    }

    None
}

fn detect_intel() -> Option<GpuInfo> {
    let has_vulkan = Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .ok()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !has_vulkan {
        return None;
    }

    for entry in std::fs::read_dir("/sys/class/drm").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains("-") {
            continue;
        }
        let vendor_path = entry.path().join("device/vendor");
        if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
            let v = vendor.trim();
            if v == "0x8086" {
                let vram = detect_intel_vram_from_sysfs().unwrap_or(8.0);
                return Some(GpuInfo {
                    has_gpu: true,
                    name: "Intel GPU (Vulkan)".to_string(),
                    vram_gb: vram,
                    backend: GpuBackend::Vulkan,
                });
            }
        }
    }

    None
}

fn detect_intel_vram_from_sysfs() -> Option<f64> {
    let vram_path = std::path::Path::new("/sys/class/drm/card0/device/mem_info_vram_total");
    if let Ok(vram_str) = std::fs::read_to_string(vram_path) {
        if let Ok(vram_bytes) = vram_str.trim().parse::<u64>() {
            return Some(vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        }
    }
    None
}

fn detect_apple() -> Option<GpuInfo> {
    if cfg!(target_os = "macos") {
        let output = Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("chipset") || stdout.contains("Metal") || stdout.contains("Apple") || stdout.contains("M1") || stdout.contains("M2") || stdout.contains("M3") || stdout.contains("M4") {
            let vram = detect_apple_vram().unwrap_or(16.0);
            let name = if stdout.contains("M4 Max") || stdout.contains("M4 Pro") || stdout.contains("M4") {
                "Apple Silicon (Vulkan)".to_string()
            } else if stdout.contains("M3 Max") || stdout.contains("M3 Pro") || stdout.contains("M3") {
                "Apple Silicon (Vulkan)".to_string()
            } else if stdout.contains("M2 Max") || stdout.contains("M2 Pro") || stdout.contains("M2") {
                "Apple Silicon (Vulkan)".to_string()
            } else if stdout.contains("M1 Max") || stdout.contains("M1 Pro") || stdout.contains("M1") {
                "Apple Silicon (Vulkan)".to_string()
            } else {
                "Apple GPU (Vulkan)".to_string()
            };
            return Some(GpuInfo {
                has_gpu: true,
                name,
                vram_gb: vram,
                backend: GpuBackend::Vulkan,
            });
        }
    }
    None
}

fn detect_apple_vram() -> Option<f64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let total_bytes: u64 = String::from_utf8_lossy(&output.stdout).trim().parse().ok()?;
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    Some(total_gb * 0.75)
}