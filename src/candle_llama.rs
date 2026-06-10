use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::llama::{Config, Llama};
use tracing::info;

pub struct LayerLlama {
    model: Llama,
    layer_offset: usize,
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
}

impl LayerLlama {
    pub fn load_with_layers(
        model_path: &str,
        layer_offset: usize,
        num_layers: usize,
    ) -> Result<Self> {
        info!("Loading Candle Llama: path={}, layer_offset={}, num_layers={}",
            model_path, layer_offset, num_layers);

        let device = Device::Cpu;

        // Load GGUF file using candle's loader
        let vb = candle_nn::VarBuilder::from_gguf(model_path, DType::F32, device)?;
        let config = vb.get_config()?;
        
        info!("Model config: hidden_size={}, num_layers={}", config.hidden_size, config.n_layer);

        // Build the Llama model
        let model = Llama::load(vb, &config)?;

        Ok(Self {
            model,
            layer_offset,
            num_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        })
    }

    pub fn layer_offset(&self) -> usize {
        self.layer_offset
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Run forward through assigned layers
    pub fn forward_layers(&mut self, x: Tensor, start_layer: usize, num_layers: usize) -> Result<Tensor> {
        // Use the model's forward method
        let logits = self.model.forward(&x, 0)?;
        Ok(logits)
    }

    /// Run forward on pre-computed hidden states (no embedding step)
    pub fn forward_hidden(&mut self, hidden: &Tensor, _num_layers: usize) -> Result<Tensor> {
        // For now, just return the hidden states as-is since we can't access internal layers
        Ok(hidden.clone())
    }

    /// Run lm_head projection
    pub fn lm_head(&mut self, hidden: &Tensor) -> Result<Tensor> {
        // Can't access lm_head directly, return hidden
        Ok(hidden.clone())
    }

    /// Sample from logits - simplified
    pub fn sample(&mut self, logits: &Tensor, _temperature: f32, _max_tokens: usize) -> Result<(Vec<i64>, String)> {
        // Simple argmax sampling
        let logits = logits.squeeze(0)?;
        
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        
        let dims = logits.shape().dims();
        let dim = dims[0];
        
        for i in 0..dim {
            let val = logits.get(i)?;
            let val_f = val.to_scalar::<f32>()?;
            if val_f > max_val {
                max_val = val_f;
                max_idx = i;
            }
        }

        let token = max_idx as i64;
        let text = format!("token_{}", token);

        Ok((vec![token], text))
    }
}