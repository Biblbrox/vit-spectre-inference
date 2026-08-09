use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, Gelu, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Device, Distribution, Int, activation::softmax},
};

use crate::kernels::Backend;

use crate::{
    norm::{DynamicERF, DynamicERFConfig},
    tokenization::embeddings::{PatchEmbedding, PatchEmbeddingConfig},
};

// ─────────────────────────────────────────────────────────────────────────────
// Window Attention with relative position bias
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct WindowAttention {
    qkv: Linear,
    attn_drop: Dropout,
    proj: Linear,
    proj_drop: Dropout,
    relative_position_bias_table: Param<Tensor<2>>,
    relative_position_index: Tensor<2, Int>,
    window_size: usize,
    num_heads: usize,
    
}

#[derive(Config, Debug)]
pub struct WindowAttentionConfig {
    embed_dim: usize,
    num_heads: usize,
    window_size: usize,
    dropout: f64,
}

impl WindowAttention {
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<3>>) -> Tensor<3> {
        let [b, n, _] = x.dims();
        let h = self.num_heads;
        let head_dim = self.relative_position_bias_table.val().dims()[1];

        // QKV projection: [B, N, E] -> [B, N, 3*E]
        let qkv = self.qkv.forward(x);
        let qkv = qkv
            .clone()
            .reshape([b, n, 3, h, head_dim])
            .swap_dims(0, 2)
            .swap_dims(1, 3);
        // [3, B, H, N, D]

        let q = qkv
            .clone()
            .slice([0..1, 0..b, 0..h, 0..n, 0..head_dim])
            .squeeze_dim(0);
        let k = qkv
            .clone()
            .slice([1..2, 0..b, 0..h, 0..n, 0..head_dim])
            .squeeze_dim(0);
        let v = qkv
            .clone()
            .slice([2..3, 0..b, 0..h, 0..n, 0..head_dim])
            .squeeze_dim(0);

        // Attention: Q @ K^T / sqrt(d)  [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
        let q = q * (1.0 / (head_dim as f64).sqrt());
        let attn = q.matmul(k.swap_dims(2, 3));

        // Add relative position bias: index into bias table
        // relative_position_index: [N, N], bias_table: [N*N, H]
        let bias = self.relative_position_bias_table.val().clone();
        let relative_position_bias = bias
            .gather(
                0,
                self.relative_position_index
                    .clone()
                    .reshape([n * n, 1])
                    .repeat_dim(1, h),
            )
            .reshape([n, n, h])
            .permute([2, 0, 1])
            .unsqueeze_dim(0);
        let attn = attn + relative_position_bias;

        // Apply mask if present (for shifted windows)
        let attn = match mask {
            Some(m) => attn + m,
            None => attn,
        };

        let attn = softmax(attn, 3);
        let attn = self.attn_drop.forward(attn);

        // attn @ V: [B, H, N, N] @ [B, H, N, D] -> [B, H, N, D]
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).reshape([b, n, h * head_dim]);

        let out = self.proj.forward(out);
        self.proj_drop.forward(out)
    }
}

impl WindowAttentionConfig {
    pub fn init(&self, device: &Device) -> WindowAttention {
        let n = self.window_size.pow(2);
        let h = self.num_heads;

        // Build relative position index
        let coords_h = Tensor::<1, Int>::arange(0..self.window_size as i64, device);
        let coords_w = Tensor::<1, Int>::arange(0..self.window_size as i64, device);

        let coords_h = coords_h.unsqueeze_dim(1).repeat_dim(1, self.window_size);
        let coords_w = coords_w.unsqueeze_dim(0).repeat_dim(0, self.window_size);

        let coords_h = coords_h.flatten(0, -1).unsqueeze_dim(2);
        let coords_w = coords_w.flatten(0, -1).unsqueeze_dim(2);
        let coords = Tensor::cat(vec![coords_h, coords_w], 2);

        let coords_flatten = coords.unsqueeze_dim(1);
        let relative_coords = coords_flatten.clone() - coords.unsqueeze_dim(0);

        let mut relative_coords = relative_coords;
        let slice_h = relative_coords.slice([0..n, 0..n, 0..1]);
        let slice_w = relative_coords.slice([0..n, 0..n, 1..2]);
        let slice_h = slice_h + (self.window_size as i64 - 1);
        let slice_w = slice_w + (self.window_size as i64 - 1);
        let slice_w = slice_w * (2 * self.window_size as i64 - 1) as f64;
        let relative_position_index = (slice_h + slice_w).squeeze_dim(2);

        // Initialize relative position bias table [2*Wh-1 * 2*Ww-1, H]
        let bias_table = Tensor::<2>::random(
            [(2 * self.window_size - 1) * (2 * self.window_size - 1), h],
            Distribution::Normal(0.0, 0.1),
            device,
        );

        WindowAttention {
            qkv: LinearConfig::new(self.embed_dim, 3 * self.embed_dim).init(device),
            attn_drop: DropoutConfig::new(self.dropout).init(),
            proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            proj_drop: DropoutConfig::new(self.dropout).init(),
            relative_position_bias_table: Param::from_tensor(bias_table).set_require_grad(true),
            relative_position_index,
            window_size: self.window_size,
            num_heads: self.num_heads,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swin Transformer Block
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct SwinBlock {
    attn: WindowAttention,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    norm1: DynamicERF,
    norm2: DynamicERF,
    dropout: Dropout,
    activation: Gelu,
    window_size: usize,
    shift_size: usize,
    
}

#[derive(Config, Debug)]
pub struct SwinBlockConfig {
    embed_dim: usize,
    num_heads: usize,
    window_size: usize,
    shift_size: usize,
    hidden_dim: usize,
    dropout: f64,
}

impl SwinBlock {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();
        let h = (n as f64).sqrt() as usize;
        let w = h;

        // Cyclic shift for shifted window attention
        let shift = (self.shift_size > 0) && (self.shift_size < self.window_size);

        let x = if shift {
            let shifted = x.reshape([b, h, w, e]);
            let shifted = shifted.roll_dim(-(self.shift_size as i64), 1);
            let shifted = shifted.roll_dim(-(self.shift_size as i64), 2);
            shifted.reshape([b, n, e])
        } else {
            x
        };

        // Window partition
        let x_windows = self.partition_windows(x.clone());

        // Window attention
        let attn_out = self.attn.forward(x_windows, None);

        // Window reverse
        let x = self.reverse_windows(attn_out, h, w);

        // Reverse cyclic shift
        let x = if shift {
            let reversed = x.reshape([b, h, w, e]);
            let reversed = reversed.roll_dim(self.shift_size as i64, 1);
            let reversed = reversed.roll_dim(self.shift_size as i64, 2);
            reversed.reshape([b, n, e])
        } else {
            x
        };

        // Residual + MLP
        let x = x.clone() + self.dropout.forward(x);
        let x_res = x.clone();

        let x = self.norm2.forward(x);
        let x = self.mlp_linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.mlp_linear2.forward(x);
        let x = self.dropout.forward(x);

        x_res + x
    }

    fn partition_windows(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();
        let h = (n as f64).sqrt() as usize;
        let w = h;
        let ws = self.window_size;

        let x = x.reshape([b, h / ws, ws, w / ws, ws, e]);
        let x = x.swap_dims(2, 3);
        x.reshape([b * (h / ws) * (w / ws), ws * ws, e])
    }

    fn reverse_windows(&self, x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
        let [b_windows, ws2, e] = x.dims();
        let b = b_windows / ((h / self.window_size) * (w / self.window_size));
        let ws = self.window_size;

        let x = x.reshape([b, h / ws, w / ws, ws, ws, e]);
        let x = x.swap_dims(2, 3);
        x.reshape([b, h * w, e])
    }
}

impl SwinBlockConfig {
    pub fn init(&self, device: &Device) -> SwinBlock {
        SwinBlock {
            attn: WindowAttentionConfig::new(
                self.embed_dim,
                self.num_heads,
                self.window_size,
                self.dropout,
            )
            .init(device),
            mlp_linear1: LinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            mlp_linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
            norm1: DynamicERFConfig::new(self.embed_dim, 0.5, 0.0).init(device),
            norm2: DynamicERFConfig::new(self.embed_dim, 0.5, 0.0).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Gelu::new(),
            window_size: self.window_size,
            shift_size: self.shift_size,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Patch Merging (downsampling)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct PatchMerging {
    norm: DynamicERF,
    linear: Linear,
    in_h: usize,
    in_w: usize,
    
}

#[derive(Config, Debug)]
pub struct PatchMergingConfig {
    in_dim: usize,
    out_dim: usize,
    in_h: usize,
    in_w: usize,
}

impl PatchMerging {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();

        // Reshape to 2D grid
        let x = x.reshape([b, self.in_h, self.in_w, e]);

        // Partition into 2x2 blocks
        let x0 = x
            .clone()
            .slice([0..b, 0..self.in_h, 0..self.in_w / 2, 0..e]);
        let x1 = x
            .clone()
            .slice([0..b, 0..self.in_h, self.in_w / 2..self.in_w, 0..e]);
        let x2 = x
            .clone()
            .slice([0..b, 0..self.in_h / 2, 0..self.in_w / 2, 0..e]);
        let x3 = x.slice([0..b, self.in_h / 2..self.in_h, 0..self.in_w / 2, 0..e]);

        // Concatenate along channel dimension
        let x = Tensor::cat(vec![x0, x1, x2, x3], 3);

        // Reshape and project
        let out_h = self.in_h / 2;
        let out_w = self.in_w / 2;
        let x = x.reshape([b, out_h * out_w, 4 * e]);

        let x = self.norm.forward(x);
        self.linear.forward(x)
    }
}

impl PatchMergingConfig {
    pub fn init(&self, device: &Device) -> PatchMerging {
        PatchMerging {
            norm: DynamicERFConfig::new(self.in_dim, 0.5, 0.0).init(device),
            linear: LinearConfig::new(4 * self.in_dim, self.out_dim).init(device),
            in_h: self.in_h,
            in_w: self.in_w,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swin Encoder (multiple stages)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct SwinEncoder {
    stages: Vec<SwinStage>,
    norm: Option<DynamicERF>,
    
}

#[derive(Module, Debug)]
struct SwinStage {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
}

#[derive(Config, Debug)]
pub struct SwinEncoderConfig {
    num_layers: Vec<usize>,
    embed_dims: Vec<usize>,
    num_heads: Vec<usize>,
    window_size: usize,
    hidden_dim: usize,
    dropout: f64,
    grid_shapes: Vec<(usize, usize)>,
}

impl SwinEncoder {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mut output = x;
        for stage in &self.stages {
            output = stage.forward(output);
        }

        if let Some(norm) = &self.norm {
            output = norm.forward(output);
        }

        output
    }
}

impl SwinStage {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mut output = x;

        // Process blocks
        for (i, block) in self.blocks.iter().enumerate() {
            // Alternate shift: even blocks no shift, odd blocks shift
            output = block.forward(output);
        }

        // Downsample at end of stage (if present)
        if let Some(downsample) = &self.downsample {
            output = downsample.forward(output);
        }

        output
    }
}

impl SwinEncoderConfig {
    pub fn init(&self, device: &Device) -> SwinEncoder {
        let num_stages = self.num_layers.len();
        let mut stages = Vec::new();

        for stage_idx in 0..num_stages {
            let embed_dim = self.embed_dims[stage_idx];
            let num_heads = self.num_heads[stage_idx];
            let num_blocks = self.num_layers[stage_idx];
            let (in_h, in_w) = self.grid_shapes[stage_idx];

            // Create blocks for this stage
            let mut blocks = Vec::new();
            for block_idx in 0..num_blocks {
                let shift_size = if block_idx % 2 == 0 {
                    0
                } else {
                    self.window_size / 2
                };

                blocks.push(
                    SwinBlockConfig::new(
                        embed_dim,
                        num_heads,
                        self.window_size,
                        shift_size,
                        self.hidden_dim,
                        self.dropout,
                    )
                    .init(device),
                );
            }

            // Downsample after stage (except last stage)
            let downsample = if stage_idx < num_stages - 1 {
                let (out_h, out_w) = self.grid_shapes[stage_idx + 1];
                let out_dim = self.embed_dims[stage_idx + 1];
                Some(PatchMergingConfig::new(embed_dim, out_dim, in_h, in_w).init(device))
            } else {
                None
            };

            stages.push(SwinStage { blocks, downsample });
        }

        SwinEncoder { stages, norm: None }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Swin Transformer (full model)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct SwinTransformer {
    embedding_block: PatchEmbedding,
    encoder: SwinEncoder,
    layer_norm: DynamicERF,
    linear: Linear,
    tok_proj: Conv1d,
    
}

#[derive(Config, Debug)]
pub struct SwinTransformerConfig {
    in_channels: usize,
    embed_dim: usize,
    num_heads: Vec<usize>,
    num_layers: Vec<usize>,
    num_classes: usize,
    patch_size: usize,
    image_size: usize,
    hidden_dim: usize,
    window_size: usize,
    dropout: f64,
}

impl SwinTransformer {
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let x = self.embedding_block.forward(images);
        let x = self.encoder.forward(x);
        let x = self.layer_norm.forward(x);

        // Classification head
        self.linear.forward(self.tok_proj.forward(x)).squeeze()
    }
}

impl SwinTransformerConfig {
    pub fn init(&self, device: &Device) -> SwinTransformer {
        let grid_size = self.image_size / self.patch_size;
        let num_patches = grid_size.pow(2);

        // Compute grid shapes for each stage (halving each time)
        let mut grid_shapes = Vec::new();
        let mut h = grid_size;
        let mut w = grid_size;

        for _ in 0..self.num_layers.len() {
            grid_shapes.push((h, w));
            h = (h / 2).max(1);
            w = (w / 2).max(1);
        }

        // Compute embed dims for each stage (doubling each time)
        let mut embed_dims = Vec::new();
        let mut dim = self.embed_dim;
        for _ in 0..self.num_layers.len() {
            embed_dims.push(dim);
            dim *= 2;
        }

        SwinTransformer {
            embedding_block: PatchEmbeddingConfig::new(
                self.in_channels,
                embed_dims[0],
                self.patch_size,
                self.image_size,
                self.dropout,
                num_patches,
                false,
            )
            .init(device),

            encoder: SwinEncoderConfig::new(
                self.num_layers.clone(),
                embed_dims,
                self.num_heads.clone(),
                self.window_size,
                self.hidden_dim,
                self.dropout,
                grid_shapes,
            )
            .init(device),

            layer_norm: DynamicERFConfig::new(
                embed_dims.last().copied().unwrap_or(self.embed_dim),
                0.5,
                0.0,
            )
            .init(device),
            linear: LinearConfig::new(
                embed_dims.last().copied().unwrap_or(self.embed_dim),
                self.num_classes,
            )
            .init(device),
            tok_proj: Conv1dConfig::new(num_patches, 1, 1).init(device),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Shape;

    const IN_CHANNELS: usize = 3;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const EMBED_DIM: usize = 64;
    const NUM_HEADS: Vec<usize> = vec![4, 8];
    const NUM_LAYERS: Vec<usize> = vec![2, 2];
    const NUM_CLASSES: usize = 10;
    const BATCH_SIZE: usize = 4;
    const HIDDEN_DIM: usize = 128;
    const WINDOW_SIZE: usize = 4;
    const DROPOUT: f64 = 0.1;

    #[test]
    fn test_swin_transformer() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        let test_image = Tensor::<4>::zeros(
            Shape::new([BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        let model = SwinTransformerConfig::new(
            IN_CHANNELS,
            EMBED_DIM,
            NUM_HEADS,
            NUM_LAYERS,
            NUM_CLASSES,
            PATCH_SIZE,
            IMG_SIZE,
            HIDDEN_DIM,
            WINDOW_SIZE,
            DROPOUT,
        )
        .init(&device);

        let output = model.forward(test_image);
        assert_eq!(output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }

    #[test]
    fn test_window_attention() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        let window_size = 4;
        let seq_len = window_size.pow(2);
        let embed_dim = 64;
        let num_heads = 4;
        let batch_size = 2;

        let x = Tensor::<3>::random(
            Shape::new([batch_size, seq_len, embed_dim]),
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let attn = WindowAttentionConfig::new(embed_dim, num_heads, window_size, 0.1).init(&device);

        let out = attn.forward(x, None);
        assert_eq!(out.shape(), Shape::new([batch_size, seq_len, embed_dim]));
    }

    #[test]
    fn test_swin_block() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        let window_size = 4;
        let seq_len = window_size.pow(2);
        let embed_dim = 64;
        let num_heads = 4;
        let batch_size = 2;
        let hidden_dim = 128;

        let x = Tensor::<3>::random(
            Shape::new([batch_size, seq_len, embed_dim]),
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let block = SwinBlockConfig::new(
            embed_dim,
            num_heads,
            window_size,
            window_size / 2,
            hidden_dim,
            0.1,
        )
        .init(&device);

        let out = block.forward(x);
        assert_eq!(out.shape(), Shape::new([batch_size, seq_len, embed_dim]));
    }

    #[test]
    fn test_patch_merging() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        let in_h = 8;
        let in_w = 8;
        let in_dim = 64;
        let out_dim = 128;
        let batch_size = 2;
        let seq_len = in_h * in_w;

        let x = Tensor::<3>::random(
            Shape::new([batch_size, seq_len, in_dim]),
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let merger = PatchMergingConfig::new(in_dim, out_dim, in_h, in_w).init(&device);

        let out = merger.forward(x);
        let out_seq = (in_h / 2) * (in_w / 2);
        assert_eq!(out.shape(), Shape::new([batch_size, out_seq, out_dim]));
    }
}
