use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    train::{ClassificationOutput, InferenceStep, TrainStep},
    backend::Backend,
    tensor::Device,
};

use crate::data::batch::Batch;

pub mod efficientvit;
pub mod fast_vit;
pub mod fast_vit3d;
pub mod vit;

/// Parameters needed to initialize a model for training or inference.
pub struct TrainConfig {
    pub in_channels: usize,
    pub image_size: usize,
    pub num_classes: usize,
}

pub trait ModelConfig {
    type ValidModel: Module + InferenceStep<Input = Batch, Output = ClassificationOutput>;
    type TrainModel: AutodiffModule
        + TrainStep<Input = Batch, Output = ClassificationOutput>
        + core::fmt::Display
        + 'static;

    fn init_training(&self, device: &Device, config: &TrainConfig) -> Self::TrainModel;
    fn init_inference(&self, device: &Device, config: &TrainConfig) -> Self::ValidModel;
}
