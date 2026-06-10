use anyhow::Result;
use tracing::info;

pub struct LayerLlama {
    hidden_size: usize,
    vocab_size: usize,
    layer_offset: usize,
    num_assigned_layers: usize,
}

impl LayerLlama {
    pub fn load_with_layers(
        model_path: &str,
        layer_offset: usize,
        num_layers: usize,
    ) -> Result<Self> {
        info!("GGUS LayerLlama stub: path={}, layers {}-{}", model_path, layer_offset, layer_offset + num_layers);

        Ok(Self {
            hidden_size: 4096,
            vocab_size: 32000,
            layer_offset,
            num_assigned_layers: num_layers,
        })
    }

    pub fn layer_offset(&self) -> usize {
        self.layer_offset
    }

    pub fn num_layers(&self) -> usize {
        self.num_assigned_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Run forward on pre-computed hidden states (stub - just passes through)
    pub fn forward_hidden(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        Ok(hidden.to_vec())
    }

    /// Run lm_head projection (stub - just passes through)
    pub fn lm_head(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        Ok(hidden.to_vec())
    }

    /// Sample from logits - simple argmax
    pub fn sample(&mut self, logits: &[f32], _temperature: f32) -> Result<(Vec<i64>, String)> {
        let vocab_size = self.vocab_size;
        
        // Argmax over vocab_size (last vocab_size elements are logits for our single token)
        let offset = logits.len() - vocab_size;
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        
        for i in 0..vocab_size {
            let val = logits[offset + i];
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        let token = max_idx as i64;
        let text = format!("token_{}", token);

        Ok((vec![token], text))
    }
}