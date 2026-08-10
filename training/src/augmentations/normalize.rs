use burn::{
    Tensor,
    tensor::{Device, TensorData},
};

use crate::augmentations::Augmentation;

#[derive(Clone)]
pub struct Normalize {
    mean: Tensor<4>,
    std: Tensor<4>,
}

impl Normalize {
    pub fn new(std: Vec<f32>, mean: Vec<f32>, device: &Device) -> Normalize {
        let std_data = TensorData::new(std.clone(), [std.len()]);
        let mean_data = TensorData::new(mean.clone(), [mean.len()]);

        Normalize {
            std: Tensor::<1>::from_data(std_data, device).reshape([1, std.len(), 1, 1]),
            mean: Tensor::<1>::from_data(mean_data, device).reshape([1, mean.len(), 1, 1]),
        }
    }
}

impl Augmentation for Normalize {
    fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        let shape = input.shape();
        let mean: Tensor<4> = self.mean.clone().expand(shape.clone());
        let std: Tensor<4> = self.std.clone().expand(shape);
        input.sub(mean).div(std)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        tensor::{Device, Shape, TensorData, Tolerance},
    };

    use crate::augmentations::{Augmentation, normalize::Normalize};

    fn device() -> Device {
        Device::flex()
    }

    #[test]
    fn test_normalize_zero_std_panics_or_handles() {
        // This test documents behavior with zero std
        // Depending on your requirements, you might want to add validation
        let device = device();
        let normalize = Normalize::new(vec![0.0], vec![0.0], &device);

        let input = Tensor::<4>::ones([1, 1, 2, 2], &device);

        // This will produce infinity values - might want to handle this case
        let output = normalize.execute(input);

        // Just verify shape is preserved even with zero std
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 2]));
    }

    #[test]
    fn test_normalize_single_channel_simple_case() {
        let device = device();
        // Normalize: (x - 1) / 2
        let normalize = Normalize::new(vec![2.0], vec![1.0], &device);

        let input = Tensor::<4>::from_data(
            TensorData::new(vec![1.0f32, 3.0, 5.0, 7.0], [1, 1, 2, 2]),
            &device,
        );

        let output = normalize.execute(input);

        // (1-1)/2=0, (3-1)/2=1, (5-1)/2=2, (7-1)/2=3
        let expected = Tensor::<4>::from_data(
            TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0], [1, 1, 2, 2]),
            &device,
        );

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }
}
