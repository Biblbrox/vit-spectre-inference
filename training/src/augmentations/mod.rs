use std::marker::PhantomData;

use burn::{
    module::Module,
    prelude::Tensor,
    tensor::{Distribution, backend::Backend},
};

pub mod builder;
pub mod cloud;
pub mod colors;
pub mod mix;
pub mod normalize;
pub mod rotation;

pub trait Augmentation<B: Backend>: Send + Sync {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4>;
}

pub struct Pipeline<B: Backend> {
    transforms: Vec<Box<dyn Augmentation<B>>>,
    ph: PhantomData<B>,
}

impl<B: Backend> Default for Pipeline<B> {
    fn default() -> Pipeline<B> {
        Pipeline {
            transforms: vec![],
            ph: PhantomData,
        }
    }
}

impl<B: Backend> Pipeline<B> {
    pub fn new(transforms: Vec<Box<dyn Augmentation<B>>>) -> Pipeline<B> {
        Pipeline {
            transforms,
            ph: PhantomData,
        }
    }

    pub fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.transforms
            .iter()
            .fold(input, |acc, tr| tr.execute(acc))
    }

    /// Prepends transforms to the front of the pipeline
    pub fn prepend(mut self, mut transforms: Vec<Box<dyn Augmentation<B>>>) -> Self {
        transforms.extend(self.transforms);
        self.transforms = transforms;
        self
    }

    /// Appends transforms to the back of the pipeline
    pub fn append(mut self, transforms: Vec<Box<dyn Augmentation<B>>>) -> Self {
        self.transforms.extend(transforms);
        self
    }
}

#[derive(Module, Debug)]
pub struct DropPath<B: Backend> {
    drop_prob: f64,
    phantom: PhantomData<B>,
}

impl<B: Backend> DropPath<B> {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            drop_prob,
            phantom: PhantomData,
        }
    }

    // Applies stochastic depth: randomly drops the residual branch per sample.
    // x:        the main path (before residual add)
    // residual: the branch output to stochastically drop
    pub fn forward(&self, x: Tensor<B, 3>, residual: Tensor<B, 3>) -> Tensor<B, 3> {
        // During inference or if drop_prob is 0 — passthrough
        if self.drop_prob == 0.0 || !B::ad_enabled(&x.device()) {
            return x + residual;
        }

        let [batch, _, _] = x.dims();
        let device = x.device();
        let keep_prob = 1.0 - self.drop_prob;

        // Per-sample binary mask: [B, 1, 1] — whole residual dropped per sample
        let mask =
            Tensor::<B, 3>::random([batch, 1, 1], Distribution::Bernoulli(keep_prob), &device)
                / keep_prob; // rescale so expectation is preserved

        x + residual * mask
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::{Flex, flex::FlexDevice},
        tensor::{Shape, TensorData, Tolerance},
    };

    use crate::augmentations::{
        Augmentation, Pipeline, colors::ColorJitter, normalize::Normalize, rotation::RandomAffine,
    };

    type B = Flex;
    type Device = FlexDevice;

    #[test]
    fn test_pipeline() {
        let device = Device::default();
        let std = vec![0.5, 0.5, 0.5];
        let mean = vec![0.5, 0.5, 0.5];

        let normalize = Box::new(Normalize::<B>::new(std, mean, &device));
        let random_rotate = Box::new(RandomAffine::<B>::new(0.5, 30.0));
        let color_jitter = Box::new(ColorJitter::<B>::new(0.4, 0.4, 0.4));

        let transforms: Vec<Box<dyn Augmentation<B>>> =
            vec![normalize, random_rotate, color_jitter];
        let pipeline = Pipeline::new(transforms);

        // Fix: Use channels-first format [batch, channels, height, width]
        let input = Tensor::<B, 4>::random(
            Shape::new([128, 3, 32, 32]), // Changed from [128, 32, 32, 3]
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let res = pipeline.execute(input);

        // Verify output shape matches input shape
        assert_eq!(res.shape(), Shape::new([128, 3, 32, 32]));
    }

    #[test]
    fn test_empty_pipeline() {
        let device = Device::default();

        let pipeline = Pipeline::<B>::default();
        let input = Tensor::<B, 4>::random(
            Shape::new([128, 3, 32, 32]),
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let res = pipeline.execute(input.clone());

        // Empty pipeline should return the input unchanged
        assert_eq!(res.shape(), input.shape());
    }

    #[test]
    fn test_pipeline_append() {
        let device = Device::default();

        let normalize = Box::new(Normalize::<B>::new(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            &device,
        ));

        let pipeline = Pipeline::<B>::default().append(vec![normalize]);

        let input = Tensor::<B, 4>::ones(Shape::new([4, 3, 16, 16]), &device);
        let res = pipeline.execute(input.clone());

        assert_eq!(res.shape(), input.shape());
    }

    #[test]
    fn test_pipeline_prepend() {
        let device = Device::default();

        let normalize1 = Box::new(Normalize::<B>::new(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            &device,
        ));

        let normalize2 = Box::new(Normalize::<B>::new(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            &device,
        ));

        let pipeline = Pipeline::<B>::new(vec![normalize1]).prepend(vec![normalize2]);

        let input = Tensor::<B, 4>::ones(Shape::new([4, 3, 16, 16]), &device);
        let res = pipeline.execute(input.clone());

        assert_eq!(res.shape(), input.shape());
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================
    #[test]
    fn test_color_jitter_with_normalize() {
        let device = Device::default();
        let jitter = ColorJitter::<B>::new(0.0, 0.0, 0.0); // Identity transform

        // Create normalize that does identity: (x - 0) / 1 = x
        let normalize = Normalize::<B>::new(vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0], &device);

        let input = Tensor::<B, 4>::from_data(
            TensorData::new(
                vec![
                    0.2f32, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7,
                ],
                [1, 3, 2, 2],
            ),
            &device,
        );

        let jittered = jitter.execute(input.clone());
        let normalized = normalize.execute(jittered);

        // Both are identity transforms, so output should equal input
        input
            .to_data()
            .assert_approx_eq(&normalized.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_color_jitter_reproducibility() {
        let device = Device::default();
        // Test that zero params gives consistent results
        let jitter1 = ColorJitter::<B>::new(0.0, 0.0, 0.0);
        let jitter2 = ColorJitter::<B>::new(0.0, 0.0, 0.0);

        let input = Tensor::<B, 4>::from_data(
            TensorData::new(
                vec![
                    0.2f32, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7,
                ],
                [1, 3, 2, 2],
            ),
            &device,
        );

        let output1 = jitter1.execute(input.clone());
        let output2 = jitter2.execute(input.clone());

        // Both should give same results (identity transform)
        output1
            .to_data()
            .assert_approx_eq(&output2.to_data(), Tolerance::<f32>::balanced());
    }
}
