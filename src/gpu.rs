use std::process::Command;

pub struct GpuInfo {
    pub has_gpu: bool,
    pub name: String,
    pub vram_gb: f64,
}

pub fn detect_gpu() -> GpuInfo {
    if let Some(info) = detect_nvidia() {
        return info;
    }
    if let Some(info) = detect_rocm() {
        return info;
    }

    GpuInfo {
        has_gpu: false,
        name: "CPU only".to_string(),
        vram_gb: 0.0,
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
    })
}

fn detect_rocm() -> Option<GpuInfo> {
    let output = Command::new("rocm-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--csv")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim().split('\n').nth(1)?.split(',').collect::<Vec<_>>();

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
    })
}