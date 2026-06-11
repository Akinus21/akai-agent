use anyhow::{bail, Result};
use ggus::{GGufReader, GGmlType, GGufMetaDataValueType};
use std::fs;
use std::collections::HashMap;
use tracing::info;

pub struct LayerLlama {
    tok_embeddings: Vec<f32>,
    layers: Vec<LayerWeights>,
    ln_f_weight: Vec<f32>,
    lm_head_weight: Vec<f32>,
    hidden_size: usize,
    vocab_size: usize,
    num_layers: usize,
    layer_offset: usize,
    num_assigned_layers: usize,
    cos_cached: Vec<f32>,
    sin_cached: Vec<f32>,
    data: Vec<u8>,
}

struct LayerWeights {
    attn_q_weight: Vec<f32>,
    attn_k_weight: Vec<f32>,
    attn_v_weight: Vec<f32>,
    attn_output_weight: Vec<f32>,
    attn_norm_weight: Vec<f32>,
    ffn_gate_weight: Vec<f32>,
    ffn_down_weight: Vec<f32>,
    ffn_up_weight: Vec<f32>,
    ffn_norm_weight: Vec<f32>,
}

impl LayerLlama {
    pub fn load_with_layers(model_path: &str, layer_offset: usize, num_layers: usize) -> Result<Self> {
        info!("Loading GGUF model: path={}, layers {}-{}", model_path, layer_offset, layer_offset + num_layers);

        let data = fs::read(model_path)?;
        let mut reader = GGufReader::new(&data);
        
        let header = reader.read_header().map_err(|e| anyhow::anyhow!("GGufReadError: {:?}", e))?;
        info!("GGUF version: {}, tensors: {}, metadata: {}", 
            header.version, header.tensor_count, header.metadata_kv_count);
        
        // Read metadata and collect config values
        let mut hidden_size = 4096usize;
        let mut vocab_size = 32000usize;
        let mut total_layers = 32usize;
        let mut rope_dim = 128usize;
        let mut rope_freq_base = 10000.0f32;
        
        for _ in 0..header.metadata_kv_count {
            let kv = reader.read_meta_kv().map_err(|e| anyhow::anyhow!("GGufReadError: {:?}", e))?;
            let key = kv.key().to_string();
            let ty = kv.ty();
            let value_bytes = kv.value_bytes();
            
            match key.as_str() {
                "llama.embedding_length" if ty == GGufMetaDataValueType::U32 => {
                    if value_bytes.len() >= 4 {
                        hidden_size = u32::from_le_bytes([value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3]]) as usize;
                    }
                }
                "llama.vocab_size" if ty == GGufMetaDataValueType::U32 => {
                    if value_bytes.len() >= 4 {
                        vocab_size = u32::from_le_bytes([value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3]]) as usize;
                    }
                }
                "llama.block_count" if ty == GGufMetaDataValueType::U32 => {
                    if value_bytes.len() >= 4 {
                        total_layers = u32::from_le_bytes([value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3]]) as usize;
                    }
                }
                "llama.rope.dimension_count" if ty == GGufMetaDataValueType::U32 => {
                    if value_bytes.len() >= 4 {
                        rope_dim = u32::from_le_bytes([value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3]]) as usize;
                    }
                }
                "llama.rope.freq_base" if ty == GGufMetaDataValueType::F32 => {
                    if value_bytes.len() >= 4 {
                        rope_freq_base = f32::from_le_bytes([value_bytes[0], value_bytes[1], value_bytes[2], value_bytes[3]]);
                    }
                }
                _ => {}
            }
        }
        
        info!("Model config: hidden={}, vocab={}, layers={}, rope_dim={}", hidden_size, vocab_size, total_layers, rope_dim);

        let (cos_cached, sin_cached) = compute_rope_cache(rope_dim, rope_freq_base, 32768);
        
        // Build tensor info map
        let mut tensor_info = HashMap::new();
        for _ in 0..header.tensor_count {
            let meta = reader.read_tensor_meta().map_err(|e| anyhow::anyhow!("GGufReadError: {:?}", e))?;
            let name = meta.name().to_string();
            let info = meta.to_info();
            let n_elements: usize = info.shape().iter().copied().product::<u64>() as usize;
            tensor_info.insert(name, (info.offset(), info.nbytes(), n_elements, info.ty()));
        }
        
        let mut layers = Vec::new();
        let start_layer = layer_offset;
        let end_layer = layer_offset + num_layers;
        
        for layer_idx in start_layer..end_layer {
            info!("Loading layer {} (global {})", layers.len(), layer_idx);
            let layer = load_layer_weights(&data, &tensor_info, layer_idx)?;
            layers.push(layer);
        }
        
        let tok_embeddings = load_tensor_f32(&data, &tensor_info, "tok_embeddings.weight")?
            .ok_or_else(|| anyhow::anyhow!("missing tok_embeddings"))?;
        let ln_f_weight = load_tensor_f32(&data, &tensor_info, "ln_f.weight")?
            .ok_or_else(|| anyhow::anyhow!("missing ln_f"))?;
        let lm_head_weight = load_tensor_f32(&data, &tensor_info, "lm_head.weight")?
            .ok_or_else(|| anyhow::anyhow!("missing lm_head"))?;
        
        info!("Model loaded: {} layers assigned", layers.len());

        Ok(Self {
            tok_embeddings,
            layers,
            ln_f_weight,
            lm_head_weight,
            hidden_size,
            vocab_size,
            num_layers: total_layers,
            layer_offset,
            num_assigned_layers: num_layers,
            cos_cached,
            sin_cached,
            data,
        })
    }

    pub fn layer_offset(&self) -> usize { self.layer_offset }
    pub fn num_layers(&self) -> usize { self.num_assigned_layers }
    pub fn hidden_size(&self) -> usize { self.hidden_size }
    pub fn vocab_size(&self) -> usize { self.vocab_size }

    pub fn forward_layers(&mut self, input: &[f32], num_tokens: usize) -> Result<Vec<f32>> {
        let mut hidden = input.to_vec();
        for layer_idx in 0..self.layers.len() {
            hidden = self.process_layer(layer_idx, &hidden, num_tokens)?;
        }
        Ok(hidden)
    }

    fn process_layer(&mut self, layer_idx: usize, input: &[f32], num_tokens: usize) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_idx];
        let hs = self.hidden_size;
        
        let input_renorm = rms_norm(input, &layer.attn_norm_weight, 1e-5);
        
        let q = matmul(&input_renorm, &layer.attn_q_weight, num_tokens, hs, hs)?;
        let k = matmul(&input_renorm, &layer.attn_k_weight, num_tokens, hs, hs)?;
        let v = matmul(&input_renorm, &layer.attn_v_weight, num_tokens, hs, hs)?;
        
        let (q_rope, k_rope) = apply_rope(&q, &k, num_tokens, &self.cos_cached, &self.sin_cached)?;
        let attn_scores = softmax_scores(&q_rope, &k_rope, num_tokens, hs)?;
        let attn_output = matmul(&attn_scores, &v, num_tokens, num_tokens, hs)?;
        let attn_result = matmul(&attn_output, &layer.attn_output_weight, num_tokens, num_tokens, hs)?;
        
        let x1 = add(input, &attn_result, num_tokens, hs)?;
        let x1_norm = rms_norm(&x1, &layer.ffn_norm_weight, 1e-5);
        
        let gate = matmul(&x1_norm, &layer.ffn_gate_weight, num_tokens, hs, hs)?;
        let up = matmul(&x1_norm, &layer.ffn_up_weight, num_tokens, hs, hs)?;
        
        let silu_gate = silu(&gate)?;
        let ffn_intermediate = hadamard(&silu_gate, &up, num_tokens, hs)?;
        
        let down = matmul(&ffn_intermediate, &layer.ffn_down_weight, num_tokens, hs, hs)?;
        let output = add(&x1, &down, num_tokens, hs)?;
        Ok(output)
    }

    pub fn project(&mut self, hidden: &[f32], num_tokens: usize) -> Result<Vec<f32>> {
        let hidden_norm = rms_norm(hidden, &self.ln_f_weight, 1e-5);
        let logits = matmul_t(&hidden_norm, &self.lm_head_weight, num_tokens, self.vocab_size)?;
        Ok(logits)
    }

    pub fn sample(&mut self, logits: &[f32], temperature: f32) -> Result<(Vec<i64>, String)> {
        let scaled: Vec<f32> = if temperature > 0.0 && (temperature - 1.0).abs() > 0.001 {
            let scale = 1.0 / temperature;
            logits.iter().map(|&v| v * scale).collect()
        } else {
            logits.to_vec()
        };
        
        let probs = softmax(&scaled)?;
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_val { max_val = p; max_idx = i; }
        }
        
        Ok((vec![max_idx as i64], format!("token_{}", max_idx)))
    }
}

fn load_tensor_f32(data: &[u8], tensor_info: &HashMap<String, (usize, usize, usize, GGmlType)>, name: &str) -> Result<Option<Vec<f32>>> {
    if let Some((offset, nbytes, n_elements, ty)) = tensor_info.get(name) {
        let tensor_data = &data[*offset..*offset + nbytes];
        
        let mut float_data = vec![0f32; *n_elements];
        
        match ty {
            GGmlType::F32 => {
                for i in 0..*n_elements {
                    float_data[i] = f32::from_le_bytes([tensor_data[i*4], tensor_data[i*4+1], tensor_data[i*4+2], tensor_data[i*4+3]]);
                }
            }
            GGmlType::F16 => {
                for i in 0..*n_elements {
                    let bits = u16::from_le_bytes([tensor_data[i*2], tensor_data[i*2+1]]);
                    float_data[i] = half::f16::from_bits(bits).to_f32();
                }
            }
            GGmlType::Q8_0 => {
                let block_size = 32;
                let num_blocks = (*n_elements + block_size - 1) / block_size;
                for bi in 0..num_blocks {
                    let block_offset = bi * (block_size + 4);
                    if block_offset + block_size + 4 > tensor_data.len() { break; }
                    let scale = f32::from_le_bytes([
                        tensor_data[block_offset],
                        tensor_data[block_offset + 1],
                        tensor_data[block_offset + 2],
                        tensor_data[block_offset + 3],
                    ]);
                    for i in 0..block_size {
                        let idx = bi * block_size + i;
                        if idx >= *n_elements { break; }
                        let val = tensor_data[block_offset + 4 + i] as i8;
                        float_data[idx] = val as f32 * scale;
                    }
                }
            }
            _ => {
                info!("Unsupported tensor type {:?} for {}, skipping", ty, name);
                return Ok(None);
            }
        }
        
        Ok(Some(float_data))
    } else {
        Ok(None)
    }
}

fn load_layer_weights(data: &[u8], tensor_info: &HashMap<String, (usize, usize, usize, GGmlType)>, layer_idx: usize) -> Result<LayerWeights> {
    let p = format!("blk.{}", layer_idx);
    
    let attn_q = load_tensor_f32(data, tensor_info, &format!("{}.attn_q.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing attn_q"))?;
    let attn_k = load_tensor_f32(data, tensor_info, &format!("{}.attn_k.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing attn_k"))?;
    let attn_v = load_tensor_f32(data, tensor_info, &format!("{}.attn_v.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing attn_v"))?;
    let attn_output = load_tensor_f32(data, tensor_info, &format!("{}.attn_output.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing attn_output"))?;
    let attn_norm = load_tensor_f32(data, tensor_info, &format!("{}.attn_norm.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing attn_norm"))?;
    let ffn_gate = load_tensor_f32(data, tensor_info, &format!("{}.ffn_gate.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing ffn_gate"))?;
    let ffn_down = load_tensor_f32(data, tensor_info, &format!("{}.ffn_down.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing ffn_down"))?;
    let ffn_up = load_tensor_f32(data, tensor_info, &format!("{}.ffn_up.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing ffn_up"))?;
    let ffn_norm = load_tensor_f32(data, tensor_info, &format!("{}.ffn_norm.weight", p))?.ok_or_else(|| anyhow::anyhow!("missing ffn_norm"))?;
    
    Ok(LayerWeights {
        attn_q_weight: attn_q, attn_k_weight: attn_k, attn_v_weight: attn_v,
        attn_output_weight: attn_output, attn_norm_weight: attn_norm,
        ffn_gate_weight: ffn_gate, ffn_down_weight: ffn_down,
        ffn_up_weight: ffn_up, ffn_norm_weight: ffn_norm,
    })
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k { sum += a[i * k + l] * b[l * n + j]; }
            result[i * n + j] = sum;
        }
    }
    Ok(result)
}

fn matmul_t(a: &[f32], b: &[f32], m: usize, n: usize) -> Result<Vec<f32>> {
    let k = b.len() / n;
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k { sum += a[i * k + l] * b[j * k + l]; }
            result[i * n + j] = sum;
        }
    }
    Ok(result)
}

fn softmax_scores(q: &[f32], k: &[f32], num_tokens: usize, hs: usize) -> Result<Vec<f32>> {
    let mut scores = vec![0.0f32; num_tokens * num_tokens];
    let scale = 1.0 / (hs as f32).sqrt();
    for i in 0..num_tokens {
        for j in 0..num_tokens {
            let mut sum = 0.0f32;
            for l in 0..hs { sum += q[i * hs + l] * k[j * hs + l]; }
            scores[i * num_tokens + j] = sum * scale;
        }
    }
    for i in 0..num_tokens {
        let offset = i * num_tokens;
        let row = &scores[offset..offset + num_tokens];
        let max_val = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
        let mut exp_sum = 0.0f32;
        for j in 0..num_tokens {
            scores[offset + j] = (scores[offset + j] - max_val).exp();
            exp_sum += scores[offset + j];
        }
        for j in 0..num_tokens { scores[offset + j] /= exp_sum; }
    }
    Ok(scores)
}

fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut sum_squares = 0.0f32;
    for &x in input { sum_squares += x * x; }
    let scale = (sum_squares / n as f32 + eps).sqrt().recip();
    input.iter().enumerate().map(|(i, &x)| x * scale * weight[i]).collect()
}

fn softmax(input: &[f32]) -> Result<Vec<f32>> {
    let n = input.len();
    let max_val = input.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    let mut exp_sum = 0.0f32;
    let mut exps = Vec::with_capacity(n);
    for &x in input {
        let e = (x - max_val).exp();
        exps.push(e);
        exp_sum += e;
    }
    Ok(exps.iter().map(|&e| e / exp_sum).collect())
}

fn silu(input: &[f32]) -> Result<Vec<f32>> {
    Ok(input.iter().map(|&x| x / (1.0 + (-x).exp())).collect())
}

fn hadamard(a: &[f32], b: &[f32], _m: usize, _n: usize) -> Result<Vec<f32>> {
    if a.len() != b.len() { bail!("hadamard size mismatch"); }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect())
}

fn add(a: &[f32], b: &[f32], _m: usize, _n: usize) -> Result<Vec<f32>> {
    if a.len() != b.len() { bail!("add size mismatch"); }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

fn compute_rope_cache(rope_dim: usize, freq_base: f32, max_seq: usize) -> (Vec<f32>, Vec<f32>) {
    let head_dim = rope_dim;
    let n_elements = max_seq * (head_dim / 2);
    let mut cos = Vec::with_capacity(n_elements);
    let mut sin = Vec::with_capacity(n_elements);
    let theta = 1.0 / freq_base.powf(2.0 / head_dim as f32);
    for pos in 0..max_seq {
        for i in 0..(head_dim / 2) {
            let angle = pos as f32 * theta.powf(i as f32);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

fn apply_rope(q: &[f32], k: &[f32], num_tokens: usize, cos: &[f32], sin: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let hs = q.len() / num_tokens;
    let half_rope = hs / 2;
    let mut q_out = q.to_vec();
    let mut k_out = k.to_vec();
    for t in 0..num_tokens {
        for i in 0..half_rope {
            let idx = t * hs + i;
            let idx_rot = t * hs + i + half_rope;
            let cos_val = cos.get(t * half_rope + i).copied().unwrap_or(0.0);
            let sin_val = sin.get(t * half_rope + i).copied().unwrap_or(0.0);
            let q0 = q[idx]; let q1 = q[idx_rot];
            q_out[idx] = q0 * cos_val - q1 * sin_val;
            q_out[idx_rot] = q0 * sin_val + q1 * cos_val;
            let k0 = k[idx]; let k1 = k[idx_rot];
            k_out[idx] = k0 * cos_val - k1 * sin_val;
            k_out[idx_rot] = k0 * sin_val + k1 * cos_val;
        }
    }
    Ok((q_out, k_out))
}