use burn::{
    Tensor,
    backend::Autodiff,
    module::Module,
    nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    tensor::{
        Int,
        backend::{AutodiffBackend, Backend},
    },
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use serde::Deserialize;

use crate::{
    data::batch::Batch,
    embeddings::vit::{PatchEmbedding, PatchEmbeddingConfig},
    encoders::fast_encoder::{FastEncoder, FastEncoderConfig},
    models::{ModelConfig, TrainConfig},
    norm::{DynamicERF, DynamicERFConfig},
};

#[derive(Module, Debug)]
pub struct FastViT<B: Backend> {
    embedding_block: PatchEmbedding<B>,
    encoder: FastEncoder<B>,
    layer_norm: DynamicERF<B>,
    linear: Linear<B>,
    in_channels: usize,
    image_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FastViTConfig {
    pub dmodel: usize,
    pub num_encoders: usize,
    pub patch_size: usize,
    pub hidden_dim: usize,
    pub dropout: f64,
    pub activation: String,
    pub nheads: usize,
}

impl<B: Backend> FastViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        let x = self.encoder.forward(x);
        let x = self.layer_norm.forward(x);

        self.linear.forward(x.mean_dim(1)).squeeze()
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

impl FastViTConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> FastViT<B> {
        let grid_size = image_size / self.patch_size;
        let num_patches = grid_size.pow(2);

        FastViT {
            embedding_block: PatchEmbeddingConfig::new(
                in_channels,
                self.dmodel,
                self.patch_size,
                image_size,
                self.dropout,
                num_patches,
                false,
            )
            .init(device),

            encoder: FastEncoderConfig::new(
                self.num_encoders,
                num_patches,
                self.dmodel,
                self.hidden_dim,
                self.dropout,
                self.nheads,
            )
            .init(device),
            layer_norm: DynamicERFConfig::new(self.dmodel).init(device),
            linear: LinearConfig::new(self.dmodel, num_classes).init(device),
            in_channels,
            image_size,
        }
    }

    pub fn model_name(&self) -> String {
        format!(
            "fast_vit-hid{}-emb{}-enc{}",
            self.hidden_dim, self.dmodel, self.num_encoders
        )
    }
}

impl<B: Backend> ModelConfig<B> for FastViTConfig {
    type TrainModel = FastViT<Autodiff<B>>;
    type ValidModel = FastViT<B>;

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

impl<B: AutodiffBackend> TrainStep for FastViT<B> {
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

impl<B: Backend> InferenceStep for FastViT<B> {
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
    use super::*;
    use crate::models::fast_vit::FastViTConfig;
    use burn::{
        backend::{Flex, flex::FlexDevice},
        tensor::Shape,
    };

    type B = Flex;
    type Device = FlexDevice;

    const IN_CHANNELS: usize = 3;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const DMODEL: usize = PATCH_SIZE.pow(2) * IN_CHANNELS;
    const NUM_HEADS: usize = 8;
    const NUM_ENCODERS: usize = 4;
    const NUM_CLASSES: usize = 10;
    const BATCH_SIZE: usize = 10;
    const HIDDEN_DIM: usize = 64;
    const DROPOUT: f64 = 0.1;

    fn test_config() -> FastViTConfig {
        FastViTConfig {
            dmodel: DMODEL,
            num_encoders: NUM_ENCODERS,
            patch_size: PATCH_SIZE,
            hidden_dim: HIDDEN_DIM,
            dropout: DROPOUT,
            activation: "gelu".to_string(),
            nheads: NUM_HEADS,
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
