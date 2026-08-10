

use burn::prelude::*;

pub trait CloudAugmentation: Send + Sync {
    fn execute(&self, input: Tensor<3>) -> Tensor<3>;
}

#[derive(Default)]
pub struct CloudPipeline {
    transforms: Vec<Box<dyn CloudAugmentation>>,
    
    
}

impl CloudPipeline {
    pub fn new(transforms: Vec<Box<dyn CloudAugmentation>>) -> CloudPipeline {
        CloudPipeline {
            transforms,
        }
    }

    pub fn execute(&self, input: Tensor<3>) -> Tensor<3> {
        if self.transforms.is_empty() {
            return input;
        }

        let mut res = self.transforms[0].execute(input);
        if self.transforms.len() == 1 {
            return res;
        }
        for tr in self.transforms.iter().skip(1) {
            res = tr.execute(res);
        }

        res
    }
}

#[derive(Clone)]
pub struct CloudNormalize {
    mean: Tensor<2>,
    std: Tensor<2>,
    
}

impl CloudNormalize {
    pub fn new(std_vals: Vec<f32>, mean_vals: Vec<f32>, device: &Device) -> Self {
        let ndim = std_vals.len();
        let std_data = burn::tensor::TensorData::new(std_vals, [ndim]);
        let mean_data = burn::tensor::TensorData::new(mean_vals, [ndim]);

        CloudNormalize {
            std: Tensor::<1>::from_data(std_data, device).reshape([1, ndim]),
            mean: Tensor::<1>::from_data(mean_data, device).reshape([1, ndim]),
        }
    }
}

impl CloudAugmentation for CloudNormalize {
    fn execute(&self, input: Tensor<3>) -> Tensor<3> {
        let [b, n, c] = input.dims();
        let mean = self.mean.clone().reshape([1, 1, c]).expand([b, n, c]);
        let std = self.std.clone().reshape([1, 1, c]).expand([b, n, c]);
        (input.clone() - mean) / std
    }
}

#[derive(Debug, Clone)]
pub struct CloudRotation {
    p: f64,
    degrees: f32,
    
    
}

impl CloudRotation {
    pub fn new(p: f64, degrees: f32) -> Self {
        CloudRotation {
            p,
            degrees,
        }
    }
}

impl CloudAugmentation for CloudRotation {
    fn execute(&self, input: Tensor<3>) -> Tensor<3> {
        if fastrand::Rng::new().f64() < self.p {
            let angle = self.degrees.to_radians();
            let cos_val = f32::cos(angle);
            let sin_val = f32::sin(angle);

            // Rotation around Z axis for XY plane (first 2 channels)
            let x = input.clone().slice(burn::tensor::s![.., .., 0..1]);
            let y = input.clone().slice(burn::tensor::s![.., .., 1..2]);
            let z = input.slice(burn::tensor::s![.., .., 2..]);

            let new_x = x.clone() * cos_val - y.clone() * sin_val;
            let new_y = x * sin_val + y * cos_val;

            Tensor::cat(vec![new_x, new_y, z], 2)
        } else {
            input
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        tensor::{Device, Shape, TensorData, Tolerance},
    };

    fn device() -> Device {
        Device::flex()
    }

    #[test]
    fn test_cloud_normalize_preserves_shape() {
        let device = device();
        let normalize = CloudNormalize::new(vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0], &device);

        let input = Tensor::<3>::ones([4, 1024, 3], &device);
        let output = normalize.execute(input.clone());

        assert_eq!(output.shape(), Shape::new([4, 1024, 3]));
    }

    #[test]
    fn test_cloud_normalize_values() {
        let device = device();
        let normalize = CloudNormalize::new(vec![2.0, 2.0, 2.0], vec![1.0, 1.0, 1.0], &device);

        let input =
            Tensor::<3>::from_data(TensorData::new(vec![3.0f32, 5.0, 7.0], [1, 1, 3]), &device);

        let output = normalize.execute(input);

        let expected =
            Tensor::<3>::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [1, 1, 3]), &device);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_empty_cloud_pipeline() {
        let device = device();
        let pipeline = CloudPipeline::default();

        let input = Tensor::<3>::ones([4, 1024, 3], &device);
        let output = pipeline.execute(input.clone());

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_cloud_rotation_probability_zero_returns_unchanged() {
        let device = device();
        let rotation = CloudRotation::new(0.0, 90.0);

        let input = Tensor::<3>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]),
            &device,
        );

        let output = rotation.execute(input.clone());

        // With p=0.0, output should equal input
        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_90_degrees() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 90.0);

        // Single point (1, 2, 3) - rotate 90° around Z
        // x' = x*cos(90) - y*sin(90) = 1*0 - 2*1 = -2
        // y' = x*sin(90) + y*cos(90) = 1*1 + 2*0 = 1
        // z' = z = 3
        let input =
            Tensor::<3>::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [1, 1, 3]), &device);

        let output = rotation.execute(input);

        let expected =
            Tensor::<3>::from_data(TensorData::new(vec![-2.0f32, 1.0, 3.0], [1, 1, 3]), &device);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_180_degrees() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 180.0);

        // Point (1, 2, 3) - rotate 180° around Z
        // x' = x*cos(180) - y*sin(180) = 1*(-1) - 2*0 = -1
        // y' = x*sin(180) + y*cos(180) = 1*0 + 2*(-1) = -2
        // z' = z = 3
        let input =
            Tensor::<3>::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [1, 1, 3]), &device);

        let output = rotation.execute(input);

        let expected = Tensor::<3>::from_data(
            TensorData::new(vec![-1.0f32, -2.0, 3.0], [1, 1, 3]),
            &device,
        );

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_45_degrees() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 45.0);

        // Point (1, 0, 5) - rotate 45° around Z
        let angle = 45.0f32.to_radians();
        let cos_val = f32::cos(angle);
        let sin_val = f32::sin(angle);

        let x = 1.0f32;
        let y = 0.0f32;
        let z = 5.0f32;

        let expected_x = x * cos_val - y * sin_val;
        let expected_y = x * sin_val + y * cos_val;

        let input = Tensor::<3>::from_data(TensorData::new(vec![x, y, z], [1, 1, 3]), &device);

        let output = rotation.execute(input);

        let expected = Tensor::<3>::from_data(
            TensorData::new(vec![expected_x, expected_y, z], [1, 1, 3]),
            &device,
        );

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_z_channel_unchanged() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 45.0);

        // Z coordinates should remain unchanged after rotation
        let input = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 10.0, // Point 1: z=10
                    3.0, 4.0, 20.0, // Point 2: z=20
                    5.0, 6.0, 30.0, // Point 3: z=30
                ],
                [1, 3, 3],
            ),
            &device,
        );

        let output = rotation.execute(input.clone());

        // Extract Z channel from output
        let z_output = output.clone().slice(burn::tensor::s![.., .., 2..]);
        let z_input = input.slice(burn::tensor::s![.., .., 2..]);

        // Z values should be unchanged
        z_output
            .to_data()
            .assert_approx_eq(&z_input.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_multiple_points() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 90.0);

        // Test with multiple points
        let input = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 5.0, // Point 1: (1, 0) -> (0, 1)
                    0.0, 1.0, 10.0, // Point 2: (0, 1) -> (-1, 0)
                    2.0, 2.0, 15.0, // Point 3: (2, 2) -> (-2, 2)
                ],
                [1, 3, 3],
            ),
            &device,
        );

        let output = rotation.execute(input);

        // After 90° rotation:
        // Point 1: (0, 1, 5)
        // Point 2: (-1, 0, 10)
        // Point 3: (-2, 2, 15)
        let expected = Tensor::<3>::from_data(
            TensorData::new(
                vec![0.0, 1.0, 5.0, -1.0, 0.0, 10.0, -2.0, 2.0, 15.0],
                [1, 3, 3],
            ),
            &device,
        );

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_batch() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 90.0);

        // Test with batch of point clouds
        let input = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    // Batch 0
                    1.0, 0.0, 5.0, 0.0, 1.0, 10.0, // Batch 1
                    2.0, 0.0, 15.0, 0.0, 2.0, 20.0,
                ],
                [2, 2, 3],
            ),
            &device,
        );

        let output = rotation.execute(input);

        // Batch 0: (1,0)->(0,1), (0,1)->(-1,0)
        // Batch 1: (2,0)->(0,2), (0,2)->(-2,0)
        let expected = Tensor::<3>::from_data(
            TensorData::new(
                vec![
                    0.0, 1.0, 5.0, -1.0, 0.0, 10.0, 0.0, 2.0, 15.0, -2.0, 0.0, 20.0,
                ],
                [2, 2, 3],
            ),
            &device,
        );

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_zero_degrees() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 0.0);

        // Zero degrees rotation should not change anything
        let input = Tensor::<3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 3, 3]),
            &device,
        );

        let output = rotation.execute(input.clone());

        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_360_degrees() {
        let device = device();
        let rotation = CloudRotation::new(1.0, 360.0);

        // 360° rotation should return to original positions
        let input = Tensor::<3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]),
            &device,
        );

        let output = rotation.execute(input.clone());

        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_cloud_rotation_negative_angle() {
        let device = device();
        // -90° rotation should be equivalent to 270° rotation
        let rotation = CloudRotation::new(1.0, -90.0);

        // Point (1, 0, 5) rotated -90° around Z
        // x' = x*cos(-90) - y*sin(-90) = 1*0 - 0*(-1) = 0
        // y' = x*sin(-90) + y*cos(-90) = 1*(-1) + 0*0 = -1
        let input =
            Tensor::<3>::from_data(TensorData::new(vec![1.0f32, 0.0, 5.0], [1, 1, 3]), &device);

        let output = rotation.execute(input);

        let expected =
            Tensor::<3>::from_data(TensorData::new(vec![0.0f32, -1.0, 5.0], [1, 1, 3]), &device);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    #[test]
    fn test_cloud_pipeline_with_rotation_and_normalize() {
        let device = device();

        let normalize = CloudNormalize::new(vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0], &device);

        let rotation = CloudRotation::new(1.0, 90.0);

        let transforms: Vec<Box<dyn CloudAugmentation>> =
            vec![Box::new(rotation), Box::new(normalize)];

        let pipeline = CloudPipeline::new(transforms);

        let input = Tensor::<3>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]),
            &device,
        );

        let output = pipeline.execute(input.clone());

        // Shape should be preserved after pipeline
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_cloud_rotation_with_pipeline() {
        let device = device();

        let rotation1 = CloudRotation::new(1.0, 90.0);
        let rotation2 = CloudRotation::new(1.0, -90.0);

        let transforms: Vec<Box<dyn CloudAugmentation>> =
            vec![Box::new(rotation1), Box::new(rotation2)];

        let pipeline = CloudPipeline::new(transforms);

        // After 90° and then -90°, should return to original
        let input =
            Tensor::<3>::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [1, 1, 3]), &device);

        let output = pipeline.execute(input.clone());

        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }
}
