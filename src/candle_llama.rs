use anyhow::{bail, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_llama::{Llama, ModelWeights};
use std::path::Path;
use tracing::{info, warn};

pub struct LayerLlama {
    model: Llama,
    layer_offset: usize,
    num_layers: usize,
    hidden_size: usize,
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
        let dtype = DType::F32;

        // Load the GGUF file
        let file = std::fs::File::open(model_path)?;
        let mut reader = std::io::BufReader::new(file);

        // For now, load the full model - we'll filter layers in forward pass
        // TODO: Implement true layer-specific tensor loading
        let vb = VarBuilder::from_gguf(model_path, &mut reader, dtype, device)?;

        // Get model config from the GGUF
        let cfg = vb.get_config::<candle_transformers::models::quantized_llama::Config>()?;

        info!("Model config: vocab_size={}, hidden_size={}, num_layers={}",
            cfg.vocab_size, cfg.hidden_size, cfg.num_layers);

        // Build the model
        let model = Llama::load(vb, &cfg)?;

        Ok(Self {
            model,
            layer_offset,
            num_layers,
            hidden_size: cfg.hidden_size,
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
        self.model.vocab_size()
    }

    /// Run tok_embeddings on input token IDs
    pub fn embed_tokens(&mut self, input_ids: &[i64]) -> Result<Tensor> {
        // Get the embedding layer
        let tok_embeddings = &mut self.model.tok_embeddings;

        // Convert to tensor
        let input = Tensor::new(input_ids, &Device::Cpu)?;

        // Run embedding
        tok_embeddings.forward(&input)
    }

    /// Run forward pass on assigned layers only
    pub fn forward_layers(
        &mut self,
        input_ids: &[i64],
        injected_hidden: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Get embeddings either from tokens or injected hidden states
        let x = if let Some(hidden) = injected_hidden {
            hidden.clone()
        } else {
            self.embed_tokens(input_ids)?
        };

        // Run forward through our assigned layers
        self.model.forward_layers(x, self.layer_offset, self.num_layers)
    }

    /// Run only our assigned layers on pre-computed hidden states
    pub fn forward_hidden(&mut self, hidden: &Tensor) -> Result<Tensor> {
        self.model.forward_layers(hidden.clone(), self.layer_offset, self.num_layers)
    }

    /// Run lm_head projection and return logits
    pub fn lm_head(&mut self, hidden: &Tensor) -> Result<Tensor> {
        self.model.lm_head.forward(hidden)
    }

    /// Sample from logits with temperature
    pub fn sample(&mut self, logits: &Tensor, temperature: f32, max_tokens: usize) -> Result<(Vec<i64>, String)> {
        use candle_transformers::models::quantized_llama::Sampling;

        let mut tokens = Vec::new();
        let vocab_size = self.vocab_size();

        // Get logits for last position
        let logits = logits.squeeze(0)?;

        // Apply temperature
        let logits = if temperature > 0.0 {
            Self::apply_temperature(&logits, temperature)?
        } else {
            logits
        };

        // Greedy sample - argmax
        let probs = candle::Tensor::softmax(&logits, 0)?;
        let max_idx = Self::argmax(&probs)?;

        tokens.push(max_idx as i64);

        // For now, just return the first token as a simple implementation
        // Full generation loop would go here
        let text = format!("token_{}", max_idx);

        Ok((tokens, text))
    }

    fn apply_temperature(logits: &Tensor, temperature: f32) -> Result<Tensor> {
        let scale = 1.0 / temperature;
        let scale_tensor = Tensor::new(scale, &Device::Cpu)?;
        candle::Tensor::mul(logits, &scale_tensor)
    }

    fn argmax(probs: &Tensor) -> Result<usize> {
        let shape = probs.shape();
        let dim = shape.dim(0)?;

        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;

        for i in 0..dim {
            let val = probs.get(i)?;
            let val_f = val.to_scalar::<f32>()?;
            if val_f > max_val {
                max_val = val_f;
                max_idx = i;
            }
        }

        Ok(max_idx)
    }
}