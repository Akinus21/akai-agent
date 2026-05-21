use anyhow::{Context, Result};
use base64::Engine;
use ed25519_dalek::{SigningKey, Signer, Signature, Verifier};
use rand::rngs::OsRng;
use std::io::{self, Write};
use crate::config;

pub fn generate_keypair() -> Result<(String, String)> {
    let mut csprng = OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();

    let private_bytes = signing_key.to_bytes();
    let public_bytes = verifying_key.to_bytes();

    let private_b64 = base64::engine::general_purpose::STANDARD.encode(private_bytes);
    let public_b64 = base64::engine::general_purpose::STANDARD.encode(public_bytes);

    Ok((private_b64, public_b64))
}

pub fn ensure_keypair() -> Result<(String, String)> {
    let priv_path = config::private_key_path();
    let pub_path = config::public_key_path();

    if priv_path.exists() && pub_path.exists() {
        let priv_b64 = std::fs::read_to_string(&priv_path)?;
        let pub_b64 = std::fs::read_to_string(&pub_path)?;
        let priv_trimmed = priv_b64.trim().to_string();
        let pub_trimmed = pub_b64.trim().to_string();
        return Ok((priv_trimmed, pub_trimmed));
    }

    println!("  Generating Ed25519 keypair...");
    let (priv_b64, pub_b64) = generate_keypair()?;

    std::fs::create_dir_all(priv_path.parent().unwrap())?;
    std::fs::write(&priv_path, &priv_b64)?;
    std::fs::write(&pub_path, &pub_b64)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&priv_path, std::fs::Permissions::from_mode(0o600))?;
    }

    println!("  Keypair saved to {}", priv_path.parent().unwrap().display());
    Ok((priv_b64, pub_b64))
}

fn signing_key_from_b64(b64: &str) -> Result<SigningKey> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .context("Failed to decode private key")?;
    let array: [u8; 32] = bytes.try_into()
        .map_err(|_| anyhow::anyhow!("Private key must be 32 bytes"))?;
    Ok(SigningKey::from_bytes(&array))
}

pub fn sign_request(private_key_b64: &str, timestamp: &str, method: &str, path: &str, body: &[u8]) -> Result<String> {
    let signing_key = signing_key_from_b64(private_key_b64)?;
    let message = format!("{}|{}|{}", timestamp, method, path);
    let mut msg_bytes = message.as_bytes().to_vec();
    msg_bytes.extend_from_slice(body);
    let signature = signing_key.sign(&msg_bytes);
    Ok(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()))
}

pub fn verify_signature(public_key_b64: &str, signature_b64: &str, timestamp: &str, method: &str, path: &str, body: &[u8]) -> Result<bool> {
    let pub_bytes = base64::engine::general_purpose::STANDARD
        .decode(public_key_b64.trim())
        .context("Failed to decode public key")?;
    let pub_array: [u8; 32] = pub_bytes.try_into()
        .map_err(|_| anyhow::anyhow!("Public key must be 32 bytes"))?;
    let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pub_array)
        .map_err(|e| anyhow::anyhow!("Invalid public key: {}", e))?;

    let sig_bytes = base64::engine::general_purpose::STANDARD
        .decode(signature_b64.trim())
        .context("Failed to decode signature")?;
    let sig_array: [u8; 64] = sig_bytes.try_into()
        .map_err(|_| anyhow::anyhow!("Signature must be 64 bytes"))?;
    let signature = Signature::from_bytes(&sig_array);

    let message = format!("{}|{}|{}", timestamp, method, path);
    let mut msg_bytes = message.as_bytes().to_vec();
    msg_bytes.extend_from_slice(body);

    match verifying_key.verify(&msg_bytes, &signature) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

pub fn load_private_key() -> Result<String> {
    let priv_path = config::private_key_path();
    if !priv_path.exists() {
        anyhow::bail!("Private key not found at {}. Run `akai-agent init` first.", priv_path.display());
    }
    Ok(std::fs::read_to_string(&priv_path)?.trim().to_string())
}

pub fn load_public_key() -> Result<String> {
    let pub_path = config::public_key_path();
    if !pub_path.exists() {
        anyhow::bail!("Public key not found at {}. Run `akai-agent init` first.", pub_path.display());
    }
    Ok(std::fs::read_to_string(&pub_path)?.trim().to_string())
}

pub fn prompt_password() -> Result<String> {
    print!("  Password (Duo 2FA may trigger): ");
    io::stdout().flush()?;
    let password = rpassword::read_password()?;
    Ok(password)
}