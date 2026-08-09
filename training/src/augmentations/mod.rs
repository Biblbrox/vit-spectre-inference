
use burn::{
    backend::Backend,
    module::Module,
    prelude::{Device, Tensor},
    tensor::Distribution,
};

pub mod builder;
pub mod cloud;
pub mod colors;
pub mod mix;
pub mod normalize;
pub mod rotation;

pub trait Augmentation: Send + Sync {
    fn execute(&self, input: Tensor<4>) -> Tensor<4>;
}

pub struct Pipeline {
    transforms: Vec<Box<dyn Augmentation>>,
    
    
}

impl Default for Pipeline {
    fn default() -> Pipeline {
        Pipeline {
            transforms: vec![],
        }
    }
}

impl Pipeline {
    pub fn new(transforms: Vec<Box<dyn Augmentation>>) -> Pipeline {
        Pipeline {
            transforms,
        }
    }

    pub fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        self.transforms
            .iter()
            .fold(input, |acc, tr| tr.execute(acc))
    }

    /// Prepends transforms to the front of the pipeline
    pub fn prepend(mut self, mut transforms: Vec<Box<dyn Augmentation>>) -> Self {
        transforms.extend(self.transforms);
        self.transforms = transforms;
        self
    }

    /// Appends transforms to the back of the pipeline
    pub fn append(mut self, transforms: Vec<Box<dyn Augmentation>>) -> Self {
        self.transforms.extend(transforms);
        self
    }
}

#[derive(Module, Debug)]
pub struct DropPath {
    drop_prob: f64,
    
}

impl DropPath {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            drop_prob,
        }
    }

    // Applies stochastic depth: randomly drops the residual branch per sample.
    // x:        the main path (before residual add)
    // residual: the branch output to stochastically drop
    pub fn forward(&self, x: Tensor<3>, residual: Tensor<3>) -> Tensor<3> {
        // During inference or if drop_prob is 0 — passthrough
        if self.drop_prob == 0.0 {
            return x + residual;
        }

        let [batch, _, _] = x.dims();
        let device = x.device();
        let keep_prob = 1.0 - self.drop_prob;

        // Per-sample binary mask: [B, 1, 1] — whole residual dropped per sample
        let mask =
            Tensor::<3>::random([batch, 1, 1], Distribution::Bernoulli(keep_prob), &device)
                / keep_prob; // rescale so expectation is preserved

        x + residual * mask
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::{Flex, flex::FlexDevice},
        Shape, TensorData, Tolerance,
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

        let normalize = Box::new(Normalize::new(std, mean, &device));
        let random_rotate = Box::new(RandomAffine::new(0.5, 30.0));
        let color_jitter = Box::new(ColorJitter::new(0.4, 0.4, 0.4));

        let transforms: Vec<Box<dyn Augmentation>> =
            vec![normalize, random_rotate, color_jitter];
        let pipeline = Pipeline::new(transforms);

        // Fix: Use channels-first format [batch, channels, height, width]
        let input = Tensor::<4>::random(
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

        let pipeline = Pipeline::default();
        let input = Tensor::<4>::random(
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

        let normalize = Box::new(Normalize::new(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            &device,
        ));

        let pipeline = Pipeline::default().append(vec![normalize]);

        let input = Tensor::<4>::ones(Shape::new([4, 3, 16, 16]), &device);
        let res = pipeline.execute(input.clone());

        assert_eq!(res.shape(), input.shape());
    }

    #[test]
    fn test_pipeline_prepend() {
        let device = Device::default();

        let normalize1 = Box::new(Normalize::new(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            &device,
        ));

        let normalize2 = Box::new(Normalize::new(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            &device,
        ));

        let pipeline = Pipeline::new(vec![normalize1]).prepend(vec![normalize2]);

        let input = Tensor::<4>::ones(Shape::new([4, 3, 16, 16]), &device);
        let res = pipeline.execute(input.clone());

        assert_eq!(res.shape(), input.shape());
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================
    #[test]
    fn test_color_jitter_with_normalize() {
        let device = Device::default();
        let jitter = ColorJitter::new(0.0, 0.0, 0.0); // Identity transform

        // Create normalize that does identity: (x - 0) / 1 = x
        let normalize = Normalize::new(vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0], &device);

        let input = Tensor::<4>::from_data(
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
        let jitter1 = ColorJitter::new(0.0, 0.0, 0.0);
        let jitter2 = ColorJitter::new(0.0, 0.0, 0.0);

        let input = Tensor::<4>::from_data(
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
