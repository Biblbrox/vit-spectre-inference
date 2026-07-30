use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    train::{ClassificationOutput, InferenceStep, TrainStep},
    tensor::backend::Backend,
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

pub trait ModelConfig<B: Backend> {
    type ValidModel: Module<B> + InferenceStep<Input = Batch<B>, Output = ClassificationOutput<B>>;
    type TrainModel: AutodiffModule<Autodiff<B>, InnerModule = Self::ValidModel>
        + TrainStep<Input = Batch<Autodiff<B>>, Output = ClassificationOutput<Autodiff<B>>>
        + core::fmt::Display
        + 'static;

    fn init_training(&self, device: &B::Device, config: &TrainConfig) -> Self::TrainModel;
    fn init_inference(&self, device: &B::Device, config: &TrainConfig) -> Self::ValidModel;
}
