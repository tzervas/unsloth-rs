//! End-to-end model quantization for ternary inference.
//!
//! Provides utilities to convert pretrained FP models to ternary format
//! for memory-efficient inference.

use super::config::TernaryConfig;
use super::linear::TernaryLinear;
use super::quantize::quantize_tensor;
use crate::error::{Result, UnslothError};
use candle_core::{Device, Tensor};
use std::collections::HashMap;

/// Statistics from model quantization.
#[derive(Debug, Clone, Default)]
pub struct QuantizationStats {
    /// Number of layers quantized to ternary
    pub layers_quantized: usize,
    /// Number of layers skipped (non-linear or below threshold)
    pub layers_skipped: usize,
    /// Total parameters in original model
    pub original_params: usize,
    /// Total parameters in quantized model (as ternary)
    pub quantized_params: usize,
    /// Original model size in bytes (FP32)
    pub original_bytes: usize,
    /// Quantized model size in bytes
    pub quantized_bytes: usize,
    /// Average sparsity across quantized layers
    pub average_sparsity: f32,
    /// Per-layer sparsity
    pub layer_sparsities: HashMap<String, f32>,
}

impl QuantizationStats {
    /// Compression ratio (original / quantized).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.quantized_bytes == 0 {
            0.0
        } else {
            self.original_bytes as f32 / self.quantized_bytes as f32
        }
    }

    /// Print summary statistics.
    pub fn print_summary(&self) {
        println!("=== Quantization Summary ===");
        println!("Layers quantized: {}", self.layers_quantized);
        println!("Layers skipped: {}", self.layers_skipped);
        println!("Original params: {}", self.original_params);
        println!("Quantized params: {}", self.quantized_params);
        println!(
            "Size reduction: {:.2}x ({:.2} MB -> {:.2} MB)",
            self.compression_ratio(),
            self.original_bytes as f64 / 1e6,
            self.quantized_bytes as f64 / 1e6
        );
        println!("Average sparsity: {:.1}%", self.average_sparsity * 100.0);
    }
}

/// Result of quantizing a single linear layer.
#[derive(Debug)]
pub struct QuantizedLayer {
    /// The ternary linear layer
    pub layer: TernaryLinear,
    /// Original layer name
    pub name: String,
    /// Sparsity of the quantized weights
    pub sparsity: f32,
}

/// Configuration for model quantization.
#[derive(Debug, Clone)]
pub struct ModelQuantizationConfig {
    /// Ternary quantization config
    pub ternary_config: TernaryConfig,
    /// Minimum layer size to quantize (skip small layers)
    pub min_layer_size: usize,
    /// Skip layers matching these patterns
    pub skip_patterns: Vec<String>,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for ModelQuantizationConfig {
    fn default() -> Self {
        Self {
            ternary_config: TernaryConfig::default(),
            min_layer_size: 1024, // Skip very small layers
            skip_patterns: vec![
                "embed".to_string(),
                "norm".to_string(),
                "lm_head".to_string(),
            ],
            verbose: false,
        }
    }
}

/// Quantize a single linear layer's weights to ternary.
///
/// # Arguments
///
/// * `weight` - Weight tensor [out_features, in_features]
/// * `bias` - Optional bias tensor [out_features]
/// * `name` - Layer name for logging
/// * `config` - Quantization configuration
/// * `_device` - Target device (unused, kept for API compatibility)
///
/// # Returns
///
/// Quantized layer and statistics, or None if layer should be skipped
pub fn quantize_linear_layer(
    weight: &Tensor,
    bias: Option<&Tensor>,
    name: &str,
    config: &ModelQuantizationConfig,
    _device: &Device,
) -> Result<Option<QuantizedLayer>> {
    let dims = weight.dims();
    if dims.len() != 2 {
        return Err(UnslothError::ShapeMismatch {
            expected: vec![0, 0], // 2D
            actual: dims.to_vec(),
        });
    }

    let (out_features, in_features) = (dims[0], dims[1]);
    let num_params = out_features * in_features;

    // Check if layer should be skipped
    if num_params < config.min_layer_size {
        if config.verbose {
            println!("Skipping {} (too small: {} params)", name, num_params);
        }
        return Ok(None);
    }

    for pattern in &config.skip_patterns {
        if name.to_lowercase().contains(&pattern.to_lowercase()) {
            if config.verbose {
                println!("Skipping {} (matches pattern: {})", name, pattern);
            }
            return Ok(None);
        }
    }

    // Quantize weights
    let (ternary_weights, _scale) = quantize_tensor(weight, &config.ternary_config)?;

    let sparsity = ternary_weights.sparsity();

    if config.verbose {
        println!(
            "Quantizing {}: [{}, {}] -> sparsity {:.1}%",
            name,
            out_features,
            in_features,
            sparsity * 100.0
        );
    }

    // Create ternary linear layer
    let layer = TernaryLinear::with_config(ternary_weights, bias.cloned(), config.ternary_config)?;

    Ok(Some(QuantizedLayer {
        layer,
        name: name.to_string(),
        sparsity,
    }))
}

/// Container for a quantized transformer model.
#[derive(Debug)]
pub struct TernaryModel {
    /// Quantized linear layers by name
    pub layers: HashMap<String, TernaryLinear>,
    /// Non-quantized layers/tensors preserved from original
    pub preserved_tensors: HashMap<String, Tensor>,
    /// Quantization statistics
    pub stats: QuantizationStats,
    /// Configuration used
    pub config: ModelQuantizationConfig,
}

impl TernaryModel {
    /// Create a new empty ternary model.
    pub fn new(config: ModelQuantizationConfig) -> Self {
        Self {
            layers: HashMap::new(),
            preserved_tensors: HashMap::new(),
            stats: QuantizationStats::default(),
            config,
        }
    }

    /// Add a quantized layer.
    pub fn add_layer(&mut self, name: String, layer: TernaryLinear, sparsity: f32) {
        let (out_features, in_features) = layer.dims();
        let num_params = out_features * in_features;

        self.stats.layers_quantized += 1;
        self.stats.quantized_params += num_params;
        self.stats.original_bytes += num_params * 4; // FP32
        self.stats.quantized_bytes += layer.memory_bytes();
        self.stats.layer_sparsities.insert(name.clone(), sparsity);

        self.layers.insert(name, layer);
    }

    /// Add a preserved (non-quantized) tensor.
    pub fn add_preserved(&mut self, name: String, tensor: Tensor) {
        let num_params = tensor.elem_count();
        self.stats.layers_skipped += 1;
        self.stats.original_params += num_params;
        self.stats.original_bytes += num_params * 4;
        self.stats.quantized_bytes += num_params * 4; // Still FP32

        self.preserved_tensors.insert(name, tensor);
    }

    /// Finalize statistics after all layers added.
    pub fn finalize_stats(&mut self) {
        self.stats.original_params += self.stats.quantized_params;

        if !self.stats.layer_sparsities.is_empty() {
            self.stats.average_sparsity = self.stats.layer_sparsities.values().sum::<f32>()
                / self.stats.layer_sparsities.len() as f32;
        }
    }

    /// Get a quantized layer by name.
    pub fn get_layer(&self, name: &str) -> Option<&TernaryLinear> {
        self.layers.get(name)
    }

    /// Get a preserved tensor by name.
    pub fn get_preserved(&self, name: &str) -> Option<&Tensor> {
        self.preserved_tensors.get(name)
    }
}

/// Quantize a collection of weight tensors into a TernaryModel.
///
/// # Arguments
///
/// * `weights` - Map of layer names to weight tensors
/// * `biases` - Optional map of layer names to bias tensors
/// * `config` - Quantization configuration
/// * `device` - Target device
///
/// # Returns
///
/// Quantized model with statistics
pub fn quantize_weights_collection(
    weights: HashMap<String, Tensor>,
    biases: HashMap<String, Tensor>,
    config: ModelQuantizationConfig,
    device: &Device,
) -> Result<TernaryModel> {
    let mut model = TernaryModel::new(config.clone());

    for (name, weight) in weights {
        let bias = biases.get(&name);

        match quantize_linear_layer(&weight, bias, &name, &config, device)? {
            Some(quantized) => {
                model.add_layer(quantized.name, quantized.layer, quantized.sparsity);
            }
            None => {
                // Preserve the original weight
                model.add_preserved(name.clone(), weight);
                if let Some(b) = bias {
                    model.add_preserved(format!("{}.bias", name), b.clone());
                }
            }
        }
    }

    model.finalize_stats();

    if config.verbose {
        model.stats.print_summary();
    }

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_stats() {
        let mut stats = QuantizationStats::default();
        stats.original_bytes = 1000;
        stats.quantized_bytes = 100;

        assert!((stats.compression_ratio() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_model_quantization_config_default() {
        let config = ModelQuantizationConfig::default();
        assert_eq!(config.min_layer_size, 1024);
        assert!(config.skip_patterns.contains(&"embed".to_string()));
    }

    #[test]
    fn test_quantize_linear_layer() -> Result<()> {
        let device = Device::Cpu;
        let config = ModelQuantizationConfig {
            min_layer_size: 0, // Don't skip
            skip_patterns: vec![],
            ..Default::default()
        };

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device)?;

        let result = quantize_linear_layer(&weight, None, "test_layer", &config, &device)?;

        assert!(result.is_some());
        let quantized = result.unwrap();
        assert_eq!(quantized.name, "test_layer");
        assert!(quantized.sparsity >= 0.0 && quantized.sparsity <= 1.0);

        Ok(())
    }

    #[test]
    fn test_skip_small_layer() -> Result<()> {
        let device = Device::Cpu;
        let config = ModelQuantizationConfig {
            min_layer_size: 10000, // Large threshold
            ..Default::default()
        };

        let weight = Tensor::randn(0.0f32, 1.0, (8, 8), &device)?;

        let result = quantize_linear_layer(&weight, None, "small_layer", &config, &device)?;

        assert!(result.is_none());

        Ok(())
    }

    #[test]
    fn test_skip_pattern() -> Result<()> {
        let device = Device::Cpu;
        let config = ModelQuantizationConfig::default();

        let weight = Tensor::randn(0.0f32, 1.0, (128, 128), &device)?;

        let result = quantize_linear_layer(&weight, None, "model.embed_tokens", &config, &device)?;

        assert!(result.is_none()); // Should skip due to "embed" pattern

        Ok(())
    }

    #[test]
    fn test_ternary_model() -> Result<()> {
        let device = Device::Cpu;
        let config = ModelQuantizationConfig {
            min_layer_size: 0,
            skip_patterns: vec![],
            verbose: false,
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            Tensor::randn(0.0f32, 1.0, (64, 128), &device)?,
        );
        weights.insert(
            "layer2".to_string(),
            Tensor::randn(0.0f32, 1.0, (128, 64), &device)?,
        );

        let model = quantize_weights_collection(weights, HashMap::new(), config, &device)?;

        assert_eq!(model.stats.layers_quantized, 2);
        assert!(model.get_layer("layer1").is_some());
        assert!(model.get_layer("layer2").is_some());

        Ok(())
    }
}
