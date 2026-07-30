use burn::{
    Tensor,
    backend::Autodiff,
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{
        Int,
        backend::{AutodiffBackend, Backend},
        s,
    },
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use serde::Deserialize;

use crate::{
    data::batch::Batch,
    embeddings::vit::{PatchEmbedding, PatchEmbeddingConfig},
    models::{ModelConfig, TrainConfig},
};

/// Standard ViT implementation with cls token and fixed
/// embed_dim
#[derive(Module, Debug)]
pub struct ViT<B: Backend> {
    embedding_block: PatchEmbedding<B>,
    encoder: TransformerEncoder<B>,
    layer_norm: LayerNorm<B>,
    linear: Linear<B>,
    in_channels: usize,
    image_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ViTConfig {
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_encoders: usize,
    pub patch_size: usize,
    pub dropout: f64,
}

impl<B: Backend> ViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        let encoder_input = TransformerEncoderInput::new(x);
        let x = self.encoder.forward(encoder_input);
        let x = self.layer_norm.forward(x);
        self.linear.forward(x.slice(s![.., 0, ..])).squeeze() // [batch_size, num_classes]
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl ViTConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> ViT<B> {
        let grid_size = image_size / self.patch_size;
        let num_patches = grid_size.pow(2);
        ViT {
            embedding_block: PatchEmbeddingConfig::new(
                in_channels,
                self.embed_dim,
                self.patch_size,
                image_size,
                self.dropout,
                num_patches,
                true,
            )
            .init(device),
            encoder: TransformerEncoderConfig::new(
                self.embed_dim,
                self.hidden_dim,
                self.num_heads,
                self.num_encoders,
            )
            .with_norm_first(true)
            .with_dropout(self.dropout)
            .init(device),
            layer_norm: LayerNormConfig::new(self.embed_dim).init(device),
            linear: LinearConfig::new(self.embed_dim, num_classes).init(device),
            in_channels,
            image_size,
        }
    }

    pub fn model_name(&self) -> String {
        format!(
            "vit-head{}-hid{}-emb{}-enc{}",
            self.num_heads, self.hidden_dim, self.embed_dim, self.num_encoders
        )
    }
}

impl<B: Backend> ModelConfig<B> for ViTConfig {
    type TrainModel = ViT<Autodiff<B>>;
    type ValidModel = ViT<B>;

    fn init_training(&self, device: &B::Device, config: &TrainConfig) -> Self::TrainModel {
        self.init(
            device,
            config.in_channels,
            config.image_size,
            config.num_classes,
        )
    }

    fn init_inference(&self, device: &B::Device, config: &TrainConfig) -> Self::ValidModel {
        self.init(
            device,
            config.in_channels,
            config.image_size,
            config.num_classes,
        )
    }
}

impl<B: AutodiffBackend> TrainStep for ViT<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
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

impl<B: Backend> InferenceStep for ViT<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> ClassificationOutput<B> {
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
    use burn::{
        backend::{Flex, flex::FlexDevice},
        tensor::Shape,
    };

    use super::*;

    type B = Flex;
    type Device = FlexDevice;

    const IN_CHANNELS: usize = 3;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const EMBED_DIM: usize = PATCH_SIZE.pow(2) * IN_CHANNELS;
    const NUM_HEADS: usize = 8;
    const NUM_ENCODERS: usize = 4;
    const NUM_CLASSES: usize = 10;
    const BATCH_SIZE: usize = 10;
    const HIDDEN_DIM: usize = 64;
    const DROPOUT: f64 = 0.1;

    fn test_config() -> ViTConfig {
        ViTConfig {
            embed_dim: EMBED_DIM,
            num_heads: NUM_HEADS,
            num_encoders: NUM_ENCODERS,
            patch_size: PATCH_SIZE,
            hidden_dim: HIDDEN_DIM,
            dropout: DROPOUT,
        }
    }

    #[test]
    fn test_vit() {
        let device = Device::default();
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );
        let model = test_config().init::<B>(&device, IN_CHANNELS, IMG_SIZE, NUM_CLASSES);
        let vit_output = model.forward(test_image);
        assert_eq!(vit_output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
