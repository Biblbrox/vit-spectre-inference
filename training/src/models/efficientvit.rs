
use burn::{
    backend::{Autodiff, AutodiffBackend, Backend},
    module::Module,
    nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    prelude::*,
    tensor::Int,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use serde::Deserialize;

use crate::{
    attention::cascadedattention::{CascadedGroupAttention, CascadedGroupAttentionConfig},
    conv::{ConvBNAct, ConvBNActConfig, MBConv, MBConvConfig},
    data::batch::Batch,
    embeddings::overlapped::{PatchEmbeddingOverlapped, PatchEmbeddingOverlappedConfig},
    models::{ModelConfig, TrainConfig},
};

#[derive(Module, Debug)]
pub struct FFN {
    fc1: ConvBNAct,
    fc2: ConvBNAct,
    
}

#[derive(Config, Debug)]
pub struct FFNConfig {
    pub dim: usize,
    #[config(default = 2)]
    pub expansion_ratio: usize,
}

impl FFNConfig {
    pub fn init(&self, device: &Device) -> FFN {
        let hidden = self.dim * self.expansion_ratio;

        FFN {
            fc1: ConvBNActConfig::new(self.dim, hidden, 1).init(device),
            fc2: ConvBNActConfig::new(hidden, self.dim, 1).init(device),
        }
    }
}

impl FFN {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let residual = x.clone();

        let x = self.fc1.forward(x);
        let x = self.fc2.forward(x);

        x + residual
    }
}

#[derive(Module, Debug)]
pub struct EfficientViTBlock {
    ffn1: FFN,
    attention: CascadedGroupAttention,
    ffn2: FFN,
    
}

impl EfficientViTBlock {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.ffn1.forward(x);
        let x = self.attention.forward(x);
        self.ffn2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct EfficientViTBlockConfig {
    pub dim: usize,
    pub num_heads: usize,
    #[config(default = 2)]
    pub ffn_expansion_ratio: usize,
}

impl EfficientViTBlockConfig {
    pub fn init(&self, device: &Device) -> EfficientViTBlock {
        EfficientViTBlock {
            ffn1: FFNConfig::new(self.dim)
                .with_expansion_ratio(self.ffn_expansion_ratio)
                .init(device),
            attention: CascadedGroupAttentionConfig::new(self.dim, self.num_heads).init(device),
            ffn2: FFNConfig::new(self.dim)
                .with_expansion_ratio(self.ffn_expansion_ratio)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DownsampleBlock {
    ffn1: FFN,
    mbconv: MBConv,
    ffn2: FFN,
    
}

#[derive(Config, Debug)]
pub struct DownsampleBlockConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 2)]
    pub ffn_expansion_ratio: usize,
    #[config(default = 4)]
    pub mbconv_expand_ratio: usize,
}

impl DownsampleBlockConfig {
    pub fn init(&self, device: &Device) -> DownsampleBlock {
        DownsampleBlock {
            ffn1: FFNConfig::new(self.in_channels)
                .with_expansion_ratio(self.ffn_expansion_ratio)
                .init(device),
            mbconv: MBConvConfig::new(self.in_channels, self.out_channels)
                .with_expand_ratio(self.mbconv_expand_ratio)
                .with_stride(2)
                .init(device),
            ffn2: FFNConfig::new(self.out_channels)
                .with_expansion_ratio(self.ffn_expansion_ratio)
                .init(device),
        }
    }
}

impl DownsampleBlock {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.ffn1.forward(x);
        let x = self.mbconv.forward(x);
        self.ffn2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct EfficientViTStage {
    blocks: Vec<EfficientViTBlock>,
    
}

impl EfficientViTStage {
    pub fn forward(&self, mut x: Tensor<4>) -> Tensor<4> {
        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        x
    }
}

#[derive(Module, Debug)]
pub struct EfficientViT {
    patch_embed: PatchEmbeddingOverlapped,

    stage1: EfficientViTStage,
    down1: DownsampleBlock,

    stage2: EfficientViTStage,
    down2: DownsampleBlock,

    stage3: EfficientViTStage,

    classifier: ConvBNAct,
    head: Linear,

    in_channels: usize,
    image_size: usize,
    
}

impl EfficientViT {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let x = self.patch_embed.forward(x);

        let x = self.stage1.forward(x);
        let x = self.down1.forward(x);

        let x = self.stage2.forward(x);
        let x = self.down2.forward(x);

        let x = self.stage3.forward(x);

        let x = self.classifier.forward(x);

        let x = x.mean_dim(2).mean_dim(3).squeeze_dim::<3>(2).squeeze_dim(2);

        self.head.forward(x)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<4>,
        targets: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EfficientViTConfig {
    pub stem_channels: usize,

    pub stage_channels: [usize; 3],
    pub stage_depths: [usize; 3],
    pub stage_heads: [usize; 3],

    pub ffn_expansion_ratio: usize,
    pub mbconv_expansion_ratio: usize,
    pub attention_kernel_size: usize,
    pub dropout: f64,
    pub adam_weight_decay: f64,
    pub adam_betas: [f64; 2],
}

impl Default for EfficientViTConfig {
    fn default() -> Self {
        Self {
            stem_channels: 32,
            stage_channels: [64, 128, 256],
            stage_depths: [2, 2, 6],
            stage_heads: [2, 4, 8],
            ffn_expansion_ratio: 2,
            mbconv_expansion_ratio: 4,
            attention_kernel_size: 3,
            dropout: 0.0,
            adam_weight_decay: 0.025,
            adam_betas: [0.9, 0.999],
        }
    }
}

impl EfficientViTConfig {
    pub fn init(
        &self,
        device: &Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> EfficientViT {
        let c1 = self.stem_channels;
        let c2 = self.stage_channels[1];
        let c3 = self.stage_channels[2];

        EfficientViT {
            patch_embed: PatchEmbeddingOverlappedConfig::new(in_channels, self.stem_channels)
                .with_stem_kernel(3)
                .with_stem_stride(2)
                .with_proj_kernel(3)
                .with_proj_stride(2)
                .init(device),

            stage1: EfficientViTStageConfig::new(c1, self.stage_depths[0], self.stage_heads[0])
                .init(device),

            down1: DownsampleBlockConfig::new(c1, c2)
                .with_ffn_expansion_ratio(self.ffn_expansion_ratio)
                .with_mbconv_expand_ratio(self.mbconv_expansion_ratio)
                .init(device),

            stage2: EfficientViTStageConfig::new(c2, self.stage_depths[1], self.stage_heads[1])
                .init(device),

            down2: DownsampleBlockConfig::new(c2, c3)
                .with_ffn_expansion_ratio(self.ffn_expansion_ratio)
                .with_mbconv_expand_ratio(self.mbconv_expansion_ratio)
                .init(device),

            stage3: EfficientViTStageConfig::new(c3, self.stage_depths[2], self.stage_heads[2])
                .init(device),

            classifier: ConvBNActConfig::new(c3, c3, 1).init(device),

            head: LinearConfig::new(c3, num_classes).init(device),

            in_channels,
            image_size,
        }
    }

    pub fn model_name(&self) -> String {
        format!(
            "efficientvit-stem{}-ch{:?}-dep{:?}",
            self.stem_channels, self.stage_channels, self.stage_depths
        )
    }
}

#[derive(Config, Debug)]
pub struct EfficientViTStageConfig {
    pub dim: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
}

impl EfficientViTStageConfig {
    pub fn init(&self, device: &Device) -> EfficientViTStage {
        let mut blocks = Vec::with_capacity(self.num_blocks);
        for _ in 0..self.num_blocks {
            blocks.push(EfficientViTBlockConfig::new(self.dim, self.num_heads).init(device));
        }

        EfficientViTStage { blocks }
    }
}

impl ModelConfig for EfficientViTConfig {
    type TrainModel = EfficientViT;
    type ValidModel = EfficientViT;

    fn init_training(&self, device: &Device, config: &TrainConfig) -> Self::TrainModel {
        self.init(device, config.in_channels, config.image_size, config.num_classes)
    }

    fn init_inference(&self, device: &Device, config: &TrainConfig) -> Self::ValidModel {
        self.init(device, config.in_channels, config.image_size, config.num_classes)
    }
}

impl TrainStep for EfficientViT {
    type Input = Batch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Batch) -> TrainOutput<ClassificationOutput> {
        let images = batch.data.clone().reshape([
            batch.batch_size(),
            self.in_channels,
            self.image_size,
            self.image_size,
        ]);
        let item = self.forward_classification(images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for EfficientViT {
    type Input = Batch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Batch) -> ClassificationOutput {
        let images = batch.data.clone().reshape([
            batch.batch_size(),
            self.in_channels,
            self.image_size,
            self.image_size,
        ]);
        self.forward_classification(images, batch.targets)
    }
}

#[cfg(test)]
mod tests {

    use burn::{backend::Flex, prelude::*};

    use crate::models::efficientvit::{
        CascadedGroupAttentionConfig, DownsampleBlockConfig, EfficientViTBlockConfig,
        EfficientViTConfig, EfficientViTStageConfig,
    };

    type B = Flex;

    const IN_CHANNELS: usize = 3;
    const IMG_SIZE: usize = 224;
    const NUM_CLASSES: usize = 1000;

    fn m0_config() -> EfficientViTConfig {
        EfficientViTConfig {
            stem_channels: 64,
            stage_channels: [64, 128, 192],
            stage_depths: [1, 2, 3],
            stage_heads: [4, 4, 4],
            ffn_expansion_ratio: 2,
            mbconv_expansion_ratio: 4,
            attention_kernel_size: 3,
            dropout: 0.0,
            adam_weight_decay: 0.025,
            adam_betas: [0.9, 0.999],
        }
    }

    #[test]
    fn cga_preserves_shape() {
        let device = Default::default();

        let cfg = CascadedGroupAttentionConfig::new(64, 4).with_token_kernel_size(3);
        let model = cfg.init(&device);

        let x = Tensor::<4>::zeros([2, 64, 28, 28], &device);
        let y = model.forward(x);

        assert_eq!(y.dims(), [2, 64, 28, 28]);
    }

    #[test]
    fn efficientvit_block_preserves_shape() {
        let device = Default::default();

        let cfg = EfficientViTBlockConfig::new(64, 4).with_ffn_expansion_ratio(2);
        let model = cfg.init(&device);

        let x = Tensor::<4>::zeros([2, 64, 28, 28], &device);
        let y = model.forward(x);

        assert_eq!(y.dims(), [2, 64, 28, 28]);
    }

    #[test]
    fn downsample_block_halves_resolution_and_changes_channels() {
        let device = Default::default();

        let cfg = DownsampleBlockConfig::new(64, 128)
            .with_ffn_expansion_ratio(2)
            .with_mbconv_expand_ratio(4);
        let model = cfg.init(&device);

        let x = Tensor::<4>::zeros([2, 64, 28, 28], &device);
        let y = model.forward(x);

        assert_eq!(y.dims(), [2, 128, 14, 14]);
    }

    #[test]
    fn stage_preserves_shape() {
        let device = Default::default();

        let cfg = EfficientViTStageConfig::new(64, 2, 4);
        let model = cfg.init(&device);

        let x = Tensor::<4>::zeros([2, 64, 28, 28], &device);
        let y = model.forward(x);

        assert_eq!(y.dims(), [2, 64, 28, 28]);
    }

    #[test]
    fn efficientvit_m0_runs_end_to_end() {
        let device = Default::default();
        let model = m0_config().init(&device, IN_CHANNELS, IMG_SIZE, NUM_CLASSES);

        let x = Tensor::<4>::zeros([2, IN_CHANNELS, IMG_SIZE, IMG_SIZE], &device);
        let y = model.forward(x);

        assert_eq!(y.dims(), [2, NUM_CLASSES]);
    }
}
