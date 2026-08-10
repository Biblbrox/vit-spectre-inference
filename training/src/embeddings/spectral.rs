use std::f32::consts::PI;

use burn::{
    config::Config,
    module::Module,
    prelude::{Device, Tensor},
    tensor::{Shape, TensorData},
};

use crate::{
    norm::{DynamicERF, DynamicERFConfig},
    spectre::layers::{DctLinear, DctLinearConfig},
};

// ── DCT helpers (CPU, runs once at init) ──────────────────────────────────────

/// Normalised DCT-II matrix, shape [n, n], row-major.
/// Row k is the k-th DCT basis vector.
fn dct1d_matrix(n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for k in 0..n {
        let scale = if k == 0 {
            (1.0 / n as f32).sqrt()
        } else {
            (2.0 / n as f32).sqrt()
        };
        for i in 0..n {
            out[k * n + i] = scale * (PI * k as f32 * (2 * i + 1) as f32 / (2.0 * n as f32)).cos();
        }
    }
    out
}

/// Returns up to `count` (k1, k2) pairs in zig-zag order over a p×p DCT grid,
/// i.e. lowest spatial frequency first — identical to the JPEG coefficient scan.
fn zigzag_indices(p: usize, count: usize) -> Vec<(usize, usize)> {
    let mut idx = Vec::with_capacity(count);
    for diag in 0..(2 * p - 1) {
        let going_up = diag % 2 == 0;
        let (mut k1, mut k2) = if going_up {
            (diag.min(p - 1), diag.saturating_sub(p - 1))
        } else {
            (diag.saturating_sub(p - 1), diag.min(p - 1))
        };
        loop {
            idx.push((k1, k2));
            if idx.len() == count {
                return idx;
            }
            if going_up {
                if k1 == 0 || k2 == p - 1 {
                    break;
                }
                k1 -= 1;
                k2 += 1;
            } else {
                if k2 == 0 || k1 == p - 1 {
                    break;
                }
                k1 += 1;
                k2 -= 1;
            }
        }
    }
    idx
}

/// Builds the fixed weight matrix of shape [embed_dim, in_channels * p²].
///
/// Layout is block-diagonal over channels: channel c owns output rows
/// [c*K .. (c+1)*K) and uses the first K = embed_dim/in_channels 2-D DCT
/// basis functions (zig-zag order) applied to its p² pixel block only.
fn build_dct_weight(
    patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
    device: &Device,
) -> Tensor<2> {
    let p = patch_size;
    let p2 = p * p;
    let total_in = in_channels * p2;

    let d = dct1d_matrix(p);
    // All p² frequencies available, zig-zag ordered
    let zz = zigzag_indices(p, p2);

    let mut w = vec![0.0f32; embed_dim * total_in];
    for out_row in 0..embed_dim {
        // Cycle through channels first, then wrap frequencies
        let c = out_row % in_channels;
        let (k1, k2) = zz[(out_row / in_channels) % p2];

        for i in 0..p {
            for j in 0..p {
                let in_col = c * p2 + i * p + j;
                w[out_row * total_in + in_col] = d[k1 * p + i] * d[k2 * p + j];
            }
        }
    }

    Tensor::<2>::from_data(
        TensorData::new(w, Shape::new([embed_dim, total_in])),
        device,
    )
}

// ── DCTPatcher ────────────────────────────────────────────────────────────────

// TODO: For now, this patcher doesn't work with layers deeper than 4
// Investigation required to fix it
#[derive(Module, Debug)]
pub struct DCTPatcher {
    /// Fixed DCT projection — [embed_dim, in_channels * patch_size²].
    dct_weight: Tensor<3>,
    patch_size: usize,

    emb_dct: DctLinear,
    tok_dct: DctLinear,
    norm_tokens: DynamicERF,
    norm_embed: DynamicERF,
    
}

#[derive(Config, Debug)]
pub struct DCTPatcherConfig {
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
    seq_length: usize,
}

impl DCTPatcher {
    /// - `images` : `[B, C, H, W]`
    /// - returns  : `[B, num_patches, embed_dim]`
    pub fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        let [batch, channels, height, width] = images.dims();
        let p = self.patch_size;
        let ph = height / p;
        let pw = width / p;

        // Reshape into patch grid, then flatten each patch to a vector.
        // [B, C, H, W] -> [B, C, ph, p, pw, p]
        let x = images.reshape([batch, channels, ph, p, pw, p]);
        // -> [B, ph, pw, C, p, p]
        let x = x.permute([0, 2, 4, 1, 3, 5]);
        // -> [B, num_patches, C·p²]
        let x = x.reshape([batch, ph * pw, channels * p * p]);

        // Single matmul — no learnable params in the backward graph.
        // [B, num_patches, C·p²] × [C·p², embed_dim] → [B, num_patches, embed_dim]
        // My trial to fix gradient propagation problem. However, it seems to be that
        // normalization doesn't help here
        let x = self
            .norm_tokens
            .forward(self.tok_dct.forward(x.transpose()))
            .transpose();
        self.norm_embed.forward(self.emb_dct.forward(x))
        //x.matmul(self.dct_weight.clone().transpose())
    }
}

impl DCTPatcherConfig {
    pub fn init(&self, device: &Device) -> DCTPatcher {
        let weight =
            build_dct_weight(self.patch_size, self.in_channels, self.embed_dim, device);

        DCTPatcher {
            dct_weight: weight.unsqueeze_dim(0),
            patch_size: self.patch_size,
            emb_dct: DctLinearConfig::new(
                self.in_channels * self.patch_size.pow(2),
                self.embed_dim,
            )
            .init(device),
            tok_dct: DctLinearConfig::new(self.seq_length, self.seq_length).init(device),
            norm_tokens: DynamicERFConfig::new(self.seq_length).init(device),
            norm_embed: DynamicERFConfig::new(self.embed_dim).init(device),
        }
    }
}
