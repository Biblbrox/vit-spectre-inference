use core::{f32, f64};

use burn::{
    backend::Backend,
    Tensor,
    tensor::{
        grid::affine_grid_2d,
        ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
    },
};

use crate::augmentations::Augmentation;

// Transform2D source code taken from https://github.com/tracel-ai/burn/blob/a8ab5b3b3201ea87b2b6c1ad25a71adf1cb66f68/crates/burn-vision/src/transform/transform2d.rs
// due to heavy dependencies of the whole burn-vision crate. TODO: make pull request to the original crate
/// 2D point transformation
///
/// Useful for resampling: rotating, scaling, translating, etc image tensors
pub struct Transform2D {
    // 2x3 transformation matrix, to be used with column vectors:
    // T(x) = Ax
    transform: [[f32; 3]; 2],
}

impl Transform2D {
    /// Transforms an image
    ///
    /// * `img` - Images tensor with shape (batch_size, channels, height, width)
    ///
    /// # Returns
    ///
    /// A tensor with the same as the input
    pub fn transform(self, img: Tensor<4>) -> Tensor<4> {
        let [batch_size, channels, height, width] = img.shape().dims();
        let transform = Tensor::<2>::from(self.transform);
        let transform = transform.reshape([1, 2, 3]).expand([batch_size, 2, 3]);
        let grid = affine_grid_2d(transform, [batch_size, channels, height, width]);

        let options = GridSampleOptions::new(InterpolateMode::Bilinear)
            .with_padding_mode(GridSamplePaddingMode::Border)
            .with_align_corners(true);
        img.grid_sample_2d(grid, options)
    }

    /// Makes a 2d transformation composed of other transformations
    pub fn composed<I: IntoIterator<Item = Self>>(transforms: I) -> Self {
        let mut result = Self::identity();
        for t in transforms.into_iter() {
            result = result.mul(t);
        }
        result
    }

    /// Multiply two affine transforms represented as 2x3 matrices
    fn mul(self, other: Transform2D) -> Transform2D {
        let mut result = [[0.0f32; 3]; 2];

        // Row 0
        result[0][0] = self.transform[0][0] * other.transform[0][0]
            + self.transform[0][1] * other.transform[1][0];
        result[0][1] = self.transform[0][0] * other.transform[0][1]
            + self.transform[0][1] * other.transform[1][1];
        result[0][2] = self.transform[0][0] * other.transform[0][2]
            + self.transform[0][1] * other.transform[1][2]
            + self.transform[0][2];

        // Row 1
        result[1][0] = self.transform[1][0] * other.transform[0][0]
            + self.transform[1][1] * other.transform[1][0];
        result[1][1] = self.transform[1][0] * other.transform[0][1]
            + self.transform[1][1] * other.transform[1][1];
        result[1][2] = self.transform[1][0] * other.transform[0][2]
            + self.transform[1][1] * other.transform[1][2]
            + self.transform[1][2];

        Transform2D { transform: result }
    }

    /// Makes an identity transform (x = Ax)
    pub fn identity() -> Self {
        Self {
            transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }

    /// Makes a [`Transform2D`] for rotating a tensor
    ///
    /// * `theta` - In radians, the rotation
    /// * `cx` - Center of rotation, x
    /// * `cy` - Center of rotation, y
    pub fn rotation(theta: f32, cx: f32, cy: f32) -> Self {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let transform = [
            [cos_theta, -sin_theta, cx - cos_theta * cx + sin_theta * cy],
            [sin_theta, cos_theta, cy - sin_theta * cx - cos_theta * cy],
        ];

        Self { transform }
    }

    /// Makes a [`Transform2D`] for scaling an image tensor
    ///
    /// * `sx` - Scale factor in the x direction
    /// * `sy` - Scale factor in the y direction
    /// * `cx` - Center of scaling, x
    /// * `cy` - Center of scaling, y
    pub fn scale(sx: f32, sy: f32, cx: f32, cy: f32) -> Self {
        let transform = [[sx, 0.0, cx - sx * cx], [0.0, sy, cy - sy * cy]];

        Self { transform }
    }

    /// Makes a [`Transform2D`] for translating an image tensor
    ///
    /// * `tx` - Translation in the x direction
    /// * `ty` - Translation in the y direction
    pub fn translation(tx: f32, ty: f32) -> Self {
        let transform = [[1.0, 0.0, tx], [0.0, 1.0, ty]];

        Self { transform }
    }

    /// Applies a general shear transformation around the image center,
    /// combining both X and Y shear.
    ///
    /// # Arguments
    /// * `shx` - Shear factor along the X-axis.
    /// * `shy` - Shear factor along the Y-axis.
    /// * `cx`, `cy` - Coordinates of the image center.
    ///
    /// # Returns
    /// * `Self` with a combined shear transform matrix.
    pub fn shear(shx: f32, shy: f32, cx: f32, cy: f32) -> Self {
        let transform = [[1.0, shx, -shx * cy], [shy, 1.0, -shy * cx]];

        Self { transform }
    }
}

#[derive(Debug, Clone)]
pub struct RandomAffine {
    p: f64,
    degrees: f32,
    
    
}

impl RandomAffine {
    pub fn new(p: f64, degrees: f32) -> RandomAffine {
        RandomAffine {
            p,
            degrees,
        }
    }
}

impl Augmentation for RandomAffine {
    fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        if fastrand::Rng::new().f64() < self.p {
            let [_, _, h, w] = input.dims();
            let cx = (w as f32 - 1.0) * 0.5;
            let cy = (h as f32 - 1.0) * 0.5;

            let rot_mat = Transform2D::rotation(self.degrees.to_radians(), cx, cy);
            return rot_mat.transform(input);
        }

        input
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone)]
pub struct RandomFlip {
    p: f64,
    orientation: Orientation,
    
    
}

impl RandomFlip {
    pub fn new(p: f64, orientation: Orientation) -> RandomFlip {
        RandomFlip {
            p,
            orientation,
        }
    }
}

impl Augmentation for RandomFlip {
    fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        if fastrand::Rng::new().f64() < self.p {
            return match self.orientation {
                Orientation::Vertical => input.flip([2]),
                Orientation::Horizontal => input.flip([3]),
            };
        }
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use burn::backend::Flex;
    use burn::backend::flex::FlexDevice;
    use burn::tensor::{Shape, Tolerance};

    type B = Flex;
    type Device = FlexDevice;

    // ============================================================================
    // RandomFlip Tests
    // ============================================================================

    #[test]
    fn test_random_flip_creation() {
        let flip_h = RandomFlip::new(0.5, Orientation::Horizontal);
        let flip_v = RandomFlip::new(0.5, Orientation::Vertical);

        assert_eq!(flip_h.orientation, Orientation::Horizontal);
        assert_eq!(flip_v.orientation, Orientation::Vertical);
    }

    #[test]
    fn test_random_flip_horizontal_preserves_shape() {
        let flip = RandomFlip::new(1.0, Orientation::Horizontal);
        let input = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 3]));
        assert_eq!(input.shape(), Shape::new([1, 1, 2, 3]));
    }

    #[test]
    fn test_random_flip_vertical_preserves_shape() {
        let flip = RandomFlip::new(1.0, Orientation::Vertical);
        let input = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 3]));
        assert_eq!(input.shape(), Shape::new([1, 1, 2, 3]));
    }

    #[test]
    fn test_random_flip_probability_one_always_preserves_shape() {
        let flip = RandomFlip::new(1.0, Orientation::Horizontal);
        let input = Tensor::<4>::from([[[[1., 2.], [3., 4.]]]]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_random_flip_with_batch_images() {
        let flip = RandomFlip::new(1.0, Orientation::Horizontal);
        let input = Tensor::<4>::from([
            [
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
            ],
            [
                [[13., 14.], [15., 16.]],
                [[17., 18.], [19., 20.]],
                [[21., 22.], [23., 24.]],
            ],
        ]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), input.shape());
    }

    // ============================================================================
    // RandomAffine Tests
    // ============================================================================

    #[test]
    fn test_random_affine_creation() {
        let affine = RandomAffine::new(0.5, 30.0);

        assert_eq!(affine.p, 0.5);
        assert_eq!(affine.degrees, 30.0);
    }

    #[test]
    fn test_random_affine_preserves_shape() {
        let affine = RandomAffine::new(1.0, 30.0);
        let input = Tensor::<4>::from([[[[1., 2.], [3., 4.]]]]);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 2]));
        assert_eq!(input.shape(), Shape::new([1, 1, 2, 2]));
    }

    #[test]
    fn test_random_affine_with_batch() {
        let affine = RandomAffine::new(1.0, 45.0);
        let input = Tensor::<4>::from([[[[1., 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([2, 1, 2, 2]));
    }

    #[test]
    fn test_random_affine_large_batch() {
        let device = Device::default();

        let affine = RandomAffine::new(1.0, 30.0);
        let input = Tensor::<4>::zeros([128, 3, 32, 32], &device);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([128, 3, 32, 32]));
    }

    // ============================================================================
    // Combined/Integration Tests
    // ============================================================================

    #[test]
    fn test_random_flip_and_affine_can_be_chained() {
        let flip = RandomFlip::new(1.0, Orientation::Horizontal);
        let affine = RandomAffine::new(1.0, 30.0);

        let input = Tensor::<4>::from([[[[1., 2.], [3., 4.]]]]);

        let flipped = flip.execute(input.clone());
        let final_output = affine.execute(flipped);

        assert_eq!(final_output.shape(), input.shape());
    }

    #[test]
    fn test_random_flip_horizontal_values() {
        let flip = RandomFlip::new(1.0, Orientation::Horizontal);

        let input = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);
        let output = flip.execute(input);

        let expected = Tensor::<4>::from([[[[3., 2., 1.], [6., 5., 4.]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_flip_vertical_values() {
        let flip = RandomFlip::new(1.0, Orientation::Vertical);

        let input = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);
        let output = flip.execute(input);

        let expected = Tensor::<4>::from([[[[4., 5., 6.], [1., 2., 3.]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_affine_rotation_45_values() {
        let affine = RandomAffine::new(1.0, 45.0);

        let input = Tensor::<4>::from([[[[1., 2.], [3., 4.]]]]);

        let output = affine.execute(input.clone());

        let expected = Tensor::<4>::from([[[[1.7500, 2.7929], [1.8358, 3.7500]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_affine_rotation_90_values() {
        let affine = RandomAffine::new(1.0, 90.0);

        let input = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]]);

        let output = affine.execute(input.clone());

        let expected = Tensor::<4>::from([[[[3., 6., 9.], [3., 6., 9.], [3., 6., 9.]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_flip_probability_zero_returns_unchanged() {
        let flip = RandomFlip::new(0.0, Orientation::Horizontal);
        let input = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);

        let output = flip.execute(input.clone());

        // With p=0.0, should never flip - output equals input
        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }
}
