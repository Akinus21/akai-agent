use crate::types::HubMessage;

pub fn encode_msg(msg: &HubMessage) -> Vec<u8> {
    let mut data = serde_json::to_vec(msg).unwrap_or_default();
    data.push(b'\n');
    data
}

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

pub fn file_sha256(path: &std::path::Path) -> Option<String> {
    use sha2::{Sha256, Digest};
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => Digest::update(&mut hasher, &buf[..n]),
            Err(_) => return None,
        }
    }
    Some(format!("{:x}", Digest::finalize(hasher)))
}