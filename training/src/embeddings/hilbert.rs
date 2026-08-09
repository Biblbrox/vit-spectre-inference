
use burn::{
    config::Config,
    module::Module,
    backend::Backend,
    nn::{
        Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    
tensor::{Device, Int, Tensor, TensorData, module::adaptive_avg_pool1d},
};

use crate::curves::SpaceCurve;

/// Recursive Hilbert Tokenizer with Fixed Output Token Count
///
/// Output: [B, target_tokens, E]
///
/// Strategy:
/// 1. Fine patches
/// 2. Hilbert reorder
/// 3. Recursive merge -> hierarchical pool
/// 4. Concatenate all levels
/// 5. Project/interpolate to fixed token count
///
/// This gives:
/// - strong locality bias
/// - hierarchy
/// - global structure
/// - transformer-compatible fixed length
#[derive(Module, Debug)]
pub struct RecursiveHilbertTokenizer {
    patch_conv: Conv2d,
    token_proj: Linear,
    curve: SpaceCurve,
    levels: usize,
    target_tokens: usize,
    
}

#[derive(Config, Debug)]
pub struct RecursiveHilbertTokenizerConfig {
    pub in_channels: usize,
    pub embed_dim: usize,
    pub patch_size: usize,
    pub levels: usize,
    pub target_tokens: usize,
    pub curve: SpaceCurve,
}

impl RecursiveHilbertTokenizerConfig {
    pub fn init(&self, device: &Device) -> RecursiveHilbertTokenizer {
        RecursiveHilbertTokenizer {
            patch_conv: Conv2dConfig::new(
                [self.in_channels, self.embed_dim],
                [self.patch_size, self.patch_size],
            )
            .with_stride([self.patch_size, self.patch_size])
            .init(device),

            // Projects token dimension after interpolation
            token_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),

            curve: self.curve,
            levels: self.levels,
            target_tokens: self.target_tokens,
        }
    }
}

impl RecursiveHilbertTokenizer {
    /// Input:
    /// [B, C, H, W]
    ///
    /// Output:
    /// [B, target_tokens, E]
    pub fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        // --------------------------------------------------
        // STEP 1: Fine patch extraction
        // [B, E, H_p, W_p]
        // --------------------------------------------------
        let x = self.patch_conv.forward(images);

        let [_, _, grid_h, grid_w] = x.dims();

        // --------------------------------------------------
        // STEP 2: Flatten to [B, N, E]
        // --------------------------------------------------
        let x = x.flatten(2, 3).swap_dims(1, 2);

        // --------------------------------------------------
        // STEP 3: Hilbert reorder
        // --------------------------------------------------
        let (_, from_curve) = self.curve.build(grid_h, grid_w);

        let device = x.device();

        let indices_data = TensorData::new(from_curve.clone(), [from_curve.len()]);

        let indices = Tensor::<1, Int>::from_data(indices_data, &device);

        let mut current = x.select(1, indices);

        // --------------------------------------------------
        // STEP 4: Recursive hierarchy
        // --------------------------------------------------
        let mut all_levels: Vec<Tensor<3>> = vec![current.clone()];

        for _ in 1..self.levels {
            let [b, n, e] = current.dims();

            if n < 2 {
                break;
            }

            let usable_n = (n / 2) * 2;

            let trimmed = current.slice([0..b, 0..usable_n, 0..e]);

            // [B, N/2, 2, E]
            let grouped = trimmed.reshape([b, usable_n / 2, 2, e]);

            // Merge neighbors along Hilbert path
            let merged = grouped.mean_dim(2).squeeze_dim(2);

            all_levels.push(merged.clone());

            current = merged;
        }

        // --------------------------------------------------
        // STEP 5: Concatenate all scales
        // [B, N_total, E]
        // --------------------------------------------------
        let hierarchical = Tensor::cat(all_levels, 1);

        // --------------------------------------------------
        // STEP 6: Adaptive token projection
        //
        // Force:
        // [B, N_total, E] -> [B, target_tokens, E]
        //
        // We treat token axis like a 1D signal.
        // --------------------------------------------------
        let [b, n_total, e] = hierarchical.dims();

        if n_total == self.target_tokens {
            return self.token_proj.forward(hierarchical);
        }

        // Transpose to [B, E, N]
        let hierarchical = hierarchical.swap_dims(1, 2);

        let projected = if n_total > self.target_tokens {
            // Downsample via adaptive average pooling
            adaptive_avg_pool1d(hierarchical, self.target_tokens)
        } else {
            // Upsample via nearest repeat
            let repeat_factor = self.target_tokens.div_ceil(n_total);

            let expanded = hierarchical.repeat_dim(2, repeat_factor);

            expanded.slice([0..b, 0..e, 0..self.target_tokens])
        };

        // Back to [B, target_tokens, E]
        let projected = projected.swap_dims(1, 2);

        // --------------------------------------------------
        // STEP 7: Final learned projection
        // --------------------------------------------------
        self.token_proj.forward(projected)
    }
}
