use burn::{
    Tensor,
    backend::Autodiff,
    module::Module,
    nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    tensor::{Int, backend::AutodiffBackend, backend::Backend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use serde::Deserialize;

use crate::{
    data::batch::Batch,
    embeddings::cloud::{CloudPatchEmbedding, CloudPatchEmbeddingConfig},
    encoders::fast_encoder::{FastEncoder, FastEncoderConfig},
    models::{ModelConfig, TrainConfig},
    norm::{DynamicERF, DynamicERFConfig},
};

/// Default number of points in the ModelNet40 dataset
const NUM_POINTS: usize = 1024;
const NUM_CHANNELS: usize = 3;

#[derive(Module, Debug)]
pub struct FastViT3D<B: Backend> {
    embedding_block: CloudPatchEmbedding<B>,
    encoder: FastEncoder<B>,
    layer_norm: DynamicERF<B>,
    linear: Linear<B>,
    num_centers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FastViT3DConfig {
    pub dmodel: usize,
    pub num_encoders: usize,
    pub hidden_dim: usize,
    pub dropout: f64,
    pub activation: String,
    pub num_centers: usize,
    pub k_neighbours: usize,
    pub density_radius: f32,
    pub nheads: usize,
}

impl<B: Backend> FastViT3D<B> {
    pub fn forward(&self, points: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(points);
        let x = self.encoder.forward(x);
        let x = self.layer_norm.forward(x);

        self.linear.forward(x.mean_dim(1)).squeeze()
    }

    pub fn forward_classification(
        &self,
        points: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(points);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl FastViT3DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device, num_classes: usize) -> FastViT3D<B> {
        FastViT3D {
            embedding_block: CloudPatchEmbeddingConfig::new(
                self.num_centers,
                self.k_neighbours,
                self.density_radius,
                self.dmodel,
                self.dropout,
                false,
            )
            .init(device),

            encoder: FastEncoderConfig::new(
                self.num_encoders,
                self.num_centers,
                self.dmodel,
                self.hidden_dim,
                self.dropout,
                self.nheads,
            )
            .init(device),

            layer_norm: DynamicERFConfig::new(self.dmodel).init(device),
            linear: LinearConfig::new(self.dmodel, num_classes).init(device),
            num_centers: self.num_centers,
        }
    }

    pub fn model_name(&self) -> String {
        format!(
            "fast_vit_cloud-hid{}-emb{}-enc{}-centers{}-kn{}",
            self.hidden_dim, self.dmodel, self.num_encoders, self.num_centers, self.k_neighbours
        )
    }
}

impl<B: Backend> ModelConfig<B> for FastViT3DConfig {
    type TrainModel = FastViT3D<Autodiff<B>>;
    type ValidModel = FastViT3D<B>;

    fn init_training(&self, device: &B::Device, config: &TrainConfig) -> Self::TrainModel {
        self.init(device, config.num_classes)
    }

    fn init_inference(&self, device: &B::Device, config: &TrainConfig) -> Self::ValidModel {
        self.init(device, config.num_classes)
    }
}

impl<B: AutodiffBackend> TrainStep for FastViT3D<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let points = batch
            .data
            .clone()
            .reshape([batch.batch_size(), NUM_POINTS, NUM_CHANNELS]);
        let item = self.forward_classification(points, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for FastViT3D<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        let points = batch
            .data
            .clone()
            .reshape([batch.batch_size(), NUM_POINTS, NUM_CHANNELS]);
        self.forward_classification(points, batch.targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Flex, flex::FlexDevice},
        tensor::Shape,
    };

    type B = Flex;
    type Device = FlexDevice;

    const NUM_CENTERS: usize = 64;
    const K_NEIGHBOURS: usize = 16;
    const DENSITY_RADIUS: f32 = 0.5;
    const DMODEL: usize = 192;
    const NUM_HEADS: usize = 3;
    const NUM_ENCODERS: usize = 6;
    const NUM_CLASSES: usize = 40;
    const BATCH_SIZE: usize = 8;
    const HIDDEN_DIM: usize = 768;
    const DROPOUT: f64 = 0.001;
    const SINKHORN_TEMP: f32 = 0.05;

    fn test_config() -> FastViT3DConfig {
        FastViT3DConfig {
            dmodel: DMODEL,
            num_encoders: NUM_ENCODERS,
            hidden_dim: HIDDEN_DIM,
            dropout: DROPOUT,
            activation: "gelu".to_string(),
            num_centers: NUM_CENTERS,
            k_neighbours: K_NEIGHBOURS,
            density_radius: DENSITY_RADIUS,
            nheads: NUM_HEADS,
        }
    }

    #[test]
    #[ignore] // argtopk not implemented on Flex (CPU) backend — requires CUDA
    fn test_fast_vit3d() {
        let device = Device::default();
        let test_points =
            Tensor::<B, 3>::zeros(Shape::new([BATCH_SIZE, NUM_POINTS, NUM_CHANNELS]), &device);
        let model = test_config().init::<B>(&device, NUM_CLASSES);
        let output = model.forward(test_points);
        assert_eq!(output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
