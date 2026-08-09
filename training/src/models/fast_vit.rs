use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    tensor::{Device, Int},
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
pub struct FastViT {
    embedding_block: PatchEmbedding,
    encoder: FastEncoder,
    layer_norm: DynamicERF,
    linear: Linear,
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

impl FastViT {
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let x = self.embedding_block.forward(images);
        let x = self.encoder.forward(x);
        let x = self.layer_norm.forward(x);

        self.linear.forward(x.mean_dim(1)).squeeze()
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

impl FastViTConfig {
    pub fn init(
        &self,
        device: &Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> FastViT {
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

impl ModelConfig for FastViTConfig {
    type TrainModel = FastViT;
    type ValidModel = FastViT;

    fn init_training(&self, device: &Device, config: &TrainConfig) -> Self::TrainModel {
        self.init(
            device,
            config.in_channels,
            config.image_size,
            config.num_classes,
        )
    }

    fn init_inference(&self, device: &Device, config: &TrainConfig) -> Self::ValidModel {
        self.init(
            device,
            config.in_channels,
            config.image_size,
            config.num_classes,
        )
    }
}

impl TrainStep for FastViT {
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

impl InferenceStep for FastViT {
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
    use super::*;
    use crate::models::fast_vit::FastViTConfig;
    use burn::{
        Shape,
        backend::{Flex, flex::FlexDevice},
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
        let test_image = Tensor::<4>::zeros(
            Shape::new([BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );
        let model = test_config().init(&device, IN_CHANNELS, IMG_SIZE, NUM_CLASSES);
        let vit_output = model.forward(test_image);
        assert_eq!(vit_output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
