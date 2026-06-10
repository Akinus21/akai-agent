use anyhow::{bail, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_llama::{Config, Llama, ModelWeights};
use std::path::Path;
use tracing::{info, warn};

pub struct LayerLlama {
    model: ModelWeights,
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
        let dtype = DType::F32;

        // Load the GGUF file using candle's VarBuilder
        let vb = VarBuilder::from_gguf(model_path, device)?;

        // Get config from the GGUF file
        let config: Config = vb.get_config()?;

        info!("Model config: vocab_size={}, hidden_size={}, num_layers={}",
            config.vocab_size, config.hidden_size, config.num_layers);

        // Build model weights
        let model = ModelWeights::load(vb, &config)?;

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

    /// Run embedding on input token IDs
    pub fn embed_tokens(&mut self, input_ids: &[i64]) -> Result<Tensor> {
        let input = Tensor::new(input_ids, &Device::Cpu)?;
        self.model.tok_embeddings.forward(&input)
    }

    /// Run forward pass through assigned layers only
    pub fn forward_layers(&mut self, x: Tensor, start_layer: usize, num_layers: usize) -> Result<Tensor> {
        // Iterate through our assigned layers
        // The model has all layers, but we only process our range
        let end_layer = start_layer + num_layers;
        
        let mut x = x;
        for layer_idx in start_layer..end_layer {
            x = self.model.layers[layer_idx].forward(&x, 0)??;
        }
        
        Ok(x)
    }

    /// Run forward on pre-computed hidden states (no embedding step)
    pub fn forward_hidden(&mut self, hidden: &Tensor, num_layers: usize) -> Result<Tensor> {
        self.forward_layers(hidden.clone(), self.layer_offset, num_layers)
    }

    /// Run lm_head projection
    pub fn lm_head(&mut self, hidden: &Tensor) -> Result<Tensor> {
        self.model.lm_head.forward(hidden)
    }

    /// Sample from logits
    pub fn sample(&mut self, logits: &Tensor, temperature: f32, _max_tokens: usize) -> Result<(Vec<i64>, String)> {
        let logits = logits.squeeze(0)?;
        
        // Apply temperature if > 0
        let logits = if temperature > 0.0 && temperature != 1.0 {
            let scale = 1.0 / temperature;
            let scale_tensor = Tensor::new(scale, &Device::Cpu)?;
            candle::Tensor::mul(&logits, &scale_tensor)?
        } else {
            logits
        };

        // Softmax for probabilities
        let probs = candle::Tensor::softmax(&logits, 0)?;

        // Argmax - find the token with highest probability
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        
        let shape = probs.shape();
        let dim = shape.elem();
        
        for i in 0..dim {
            let val = probs.get(i)?;
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