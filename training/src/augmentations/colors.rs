
use burn::{
    backend::Backend,
    Tensor,
    tensor::{module::conv2d, ops::PaddedConvOptions, Device, Shape},
};

use crate::augmentations::Augmentation;

pub struct ColorJitter {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    
    
}

pub struct RandomGrayscale {
    p: f64,
    
    
}

impl ColorJitter {
    pub fn new(brightness: f32, contrast: f32, saturation: f32) -> ColorJitter {
        ColorJitter {
            brightness,
            contrast,
            saturation,
        }
    }
}

impl RandomGrayscale {
    pub fn new(p: f64) -> RandomGrayscale {
        RandomGrayscale { p }
    }
}

impl Augmentation for ColorJitter {
    // Input shape: [B, C, H, W]
    fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        let shape = input.shape();

        let mut rng = fastrand::Rng::new();
        let br =
            [1.0 + (2.0 * self.brightness as f64 * rng.f64() - self.brightness as f64) as f32; 1];
        let ctr = [1.0 + (2.0 * self.contrast as f64 * rng.f64() - self.contrast as f64) as f32; 1];
        let st =
            [1.0 + (2.0 * self.saturation as f64 * rng.f64() - self.saturation as f64) as f32; 1];

        let brightness = Tensor::<1>::from_floats(br, &input.device()).reshape([1, 1, 1, 1]);
        let contrast = Tensor::<1>::from_floats(ctr, &input.device()).reshape([1, 1, 1, 1]);
        let saturation = Tensor::<1>::from_floats(st, &input.device()).reshape([1, 1, 1, 1]);

        let brightness = brightness
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));
        let contrast: Tensor<4> = contrast
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));
        let saturation: Tensor<4> = saturation
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));

        // Adjust brightness
        let mut res: Tensor<4> = (input.clone() * brightness).clamp(0.0_f32, 1.0_f32);

        // Adjust contrast
        let mean: f32 = input.clone().mean().into_scalar();
        let adjusted: Tensor<4> = (res - mean) * contrast.clone() + mean;
        res = adjusted.clamp(0.0_f32, 1.0_f32);

        // Adjust saturation
        let r = res
            .clone()
            .slice([0..shape[0], 0..1, 0..shape[2], 0..shape[3]]);
        let g = res
            .clone()
            .slice([0..shape[0], 1..2, 0..shape[2], 0..shape[3]]);
        let b = res
            .clone()
            .slice([0..shape[0], 2..3, 0..shape[2], 0..shape[3]]);

        let gray = r.clone() * 0.2989 + g.clone() * 0.5870 + b.clone() * 0.1140;

        let r_new = ((r - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);
        let g_new = ((g - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);
        let b_new = ((b - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);

        Tensor::cat(vec![r_new, g_new, b_new], 1)
    }
}

impl Augmentation for RandomGrayscale {
    // Input shape: [B, C, H, W]
    fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        let shape = input.shape();

        if fastrand::Rng::new().f64() < self.p {
            let r = input
                .clone()
                .slice([0..shape[0], 0..1, 0..shape[2], 0..shape[3]]);
            let g = input
                .clone()
                .slice([0..shape[0], 1..2, 0..shape[2], 0..shape[3]]);
            let b = input
                .clone()
                .slice([0..shape[0], 2..3, 0..shape[2], 0..shape[3]]);

            let gray = r.clone() * 0.2989 + g.clone() * 0.5870 + b.clone() * 0.1140;
            return Tensor::cat(vec![gray.clone(), gray.clone(), gray], 1);
        }
        input
    }
}

/// Random Erasing (Zhong et al. 2020)
pub struct RandomErasing {
    p: f64,
    min_scale: f64,
    max_scale: f64,
    min_ratio: f64,
    max_ratio: f64,
    fill_value: f32,
    max_attempts: usize,
    
    
}

impl RandomErasing {
    pub fn new() -> Self {
        Self {
            p: 0.5,
            min_scale: 0.02,
            max_scale: 0.33,
            min_ratio: 0.3,
            max_ratio: 3.3,
            fill_value: 0.0,
            max_attempts: 10,
        }
    }

    pub fn with_p(mut self, p: f64) -> Self {
        self.p = p;
        self
    }

    pub fn with_scale(mut self, min: f64, max: f64) -> Self {
        self.min_scale = min;
        self.max_scale = max;
        self
    }

    pub fn with_ratio(mut self, min: f64, max: f64) -> Self {
        self.min_ratio = min;
        self.max_ratio = max;
        self
    }

    pub fn with_fill(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }
}

impl Default for RandomErasing {
    fn default() -> Self {
        Self::new()
    }
}

impl Augmentation for RandomErasing {
    // input: [B, C, H, W]
    fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        if fastrand::f64() > self.p as f64 {
            return input;
        }

        let [b, c, h, w] = [
            input.dims()[0],
            input.dims()[1],
            input.dims()[2],
            input.dims()[3],
        ];
        let area = (h * w) as f64;
        let device = input.device();

        // Find a valid erase rectangle
        let region = (0..self.max_attempts).find_map(|_| {
            let scale = (self.max_scale - self.min_scale) * fastrand::f64() + self.min_scale;
            let ratio = (self.max_ratio - self.min_ratio) * fastrand::f64() + self.min_ratio;
            let erase_area = area * scale;

            let eh = ((erase_area / ratio).sqrt() as usize).min(h);
            let ew = ((erase_area * ratio).sqrt() as usize).min(w);

            if eh == 0 || ew == 0 || eh >= h || ew >= w {
                return None;
            }
            let top = fastrand::usize(0..h - eh);
            let left = fastrand::usize(0..w - ew);
            Some((top, left, eh, ew))
        });

        let (top, left, eh, ew) = match region {
            Some(r) => r,
            None => return input,
        };

        let mut mask_data = vec![1f32; b * c * h * w];
        for bi in 0..b {
            for ci in 0..c {
                for hi in top..top + eh {
                    for wi in left..left + ew {
                        let idx = bi * (c * h * w) + ci * (h * w) + hi * w + wi;
                        mask_data[idx] = 0.0;
                    }
                }
            }
        }

        let mask = Tensor::<1>::from_floats(mask_data.as_slice(), &device).reshape([b, c, h, w]);

        let fill = Tensor::<4>::full([b, c, h, w], self.fill_value as f64, &device);

        input * mask.clone() + fill * (mask.neg() + 1.0)
    }
}

pub struct GaussianBlur {
    kernel_size: usize, // must be odd
    min_sigma: f64,
    max_sigma: f64,
    p: f64,
    device: Device,
    
    
}

impl GaussianBlur {
    pub fn new(kernel_size: usize, device: &Device) -> Self {
        assert!(kernel_size % 2 == 1, "kernel_size must be odd");
        Self {
            kernel_size,
            min_sigma: 0.1,
            max_sigma: 2.0,
            p: 0.5,
            device: device.clone(),
        }
    }

    pub fn with_sigma(mut self, min: f64, max: f64) -> Self {
        self.min_sigma = min;
        self.max_sigma = max;
        self
    }

    pub fn with_p(mut self, p: f64) -> Self {
        self.p = p;
        self
    }

    fn make_kernel(&self, channels: usize, sigma: f64) -> Tensor<4> {
        let k = self.kernel_size as i32;
        let half = k / 2;
        let mut data = vec![0f32; (k * k) as usize];
        let s2 = (sigma * sigma) as f32;

        let mut sum = 0f32;
        for ky in 0..k {
            for kx in 0..k {
                let dy = (ky - half) as f32;
                let dx = (kx - half) as f32;
                let v = (-(dx * dx + dy * dy) / (2.0 * s2)).exp();
                data[(ky * k + kx) as usize] = v;
                sum += v;
            }
        }
        for v in data.iter_mut() {
            *v /= sum;
        }

        let base = Tensor::<1>::from_floats(data.as_slice(), &self.device).reshape([
            1,
            1,
            self.kernel_size,
            self.kernel_size,
        ]);

        // repeat across channel dimension
        base.repeat(&[channels, 1, 1, 1])
    }
}

impl Augmentation for GaussianBlur {
    // input: [B, C, H, W]
    fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        if fastrand::f64() > self.p as f64 {
            return input;
        }

        let sigma = (self.max_sigma - self.min_sigma) * fastrand::f64() + self.min_sigma;
        let c = input.dims()[1];

        let kernel = self.make_kernel(c, sigma); // [C, 1, k, k]
        let pad = self.kernel_size / 2;

        // Depthwise convolution: groups = C keeps each channel independent
        // PaddedConvOptions::asymmetric(stride, padding_left, padding_right, dilation, groups)
        let options = PaddedConvOptions::asymmetric(
            [1, 1],
            [pad, pad],
            [pad, pad],
            [1, 1],
            c,
        );

        // High-level conv2d API
        conv2d(input, kernel, None, options)
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
        Augmentation,
        colors::{ColorJitter, RandomGrayscale},
    };

    type B = Flex;
    type Device = FlexDevice;

    #[test]
    fn test_color_jitter_zero_params_preserves_input() {
        let device = Device::default();
        // With all parameters set to 0, no change should occur
        let jitter = ColorJitter::new(0.0, 0.0, 0.0);

        let input = Tensor::<4>::from_data(
            TensorData::new(
                vec![
                    0.2f32, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7,
                ],
                [1, 3, 2, 2],
            ),
            &device,
        );

        let output = jitter.execute(input.clone());

        // With all factors = 1.0, output should equal input
        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_color_jitter_preserves_shape() {
        let device = Device::default();
        let jitter = ColorJitter::new(0.5, 0.5, 0.5);

        let input = Tensor::<4>::random(
            Shape::new([4, 3, 32, 32]),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let output = jitter.execute(input.clone());

        assert_eq!(output.shape(), input.shape());
        assert_eq!(output.shape(), Shape::new([4, 3, 32, 32]));
    }

    #[test]
    fn test_color_jitter_preserves_shape_batch() {
        let device = Device::default();
        let jitter = ColorJitter::new(0.3, 0.3, 0.3);

        let input = Tensor::<4>::random(
            Shape::new([16, 3, 64, 64]),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let output = jitter.execute(input.clone());

        assert_eq!(output.shape(), Shape::new([16, 3, 64, 64]));
    }

    #[test]
    fn test_color_jitter_values_in_range() {
        let device = Device::default();
        let jitter = ColorJitter::new(0.5, 0.5, 0.5);

        let input = Tensor::<4>::random(
            Shape::new([2, 3, 8, 8]),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let output = jitter.execute(input);

        // All values should be clamped to [0, 1]
        let output_data = output.to_data();
        let values: Vec<f32> = output_data.as_slice::<f32>().unwrap().to_vec();

        for &v in &values {
            assert!((0.0..=1.0).contains(&v), "Value {} out of [0, 1] range", v);
        }
    }

    #[test]
    fn test_color_jitter_different_input_shapes() {
        let device = Device::default();
        let jitter = ColorJitter::new(0.3, 0.3, 0.3);

        // Test with non-square images
        let input = Tensor::<4>::random(
            Shape::new([2, 3, 64, 128]),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let output = jitter.execute(input);
        assert_eq!(output.shape(), Shape::new([2, 3, 64, 128]));

        // Test with single pixel
        let input_small = Tensor::<4>::ones(Shape::new([1, 3, 1, 1]), &device);
        let output_small = jitter.execute(input_small);
        assert_eq!(output_small.shape(), Shape::new([1, 3, 1, 1]));
    }

    // ============================================================================
    // RandomGrayscale Tests
    // ============================================================================
    //
    #[test]
    fn test_random_grayscale_probability_one_always_grayscales() {
        let device = Device::default();
        let gray = RandomGrayscale::new(1.0);

        // Input with different channel values
        let input = Tensor::<4>::from_data(
            TensorData::new(
                vec![
                    1.0f32, 2.0, // Channel 0 (R)
                    3.0, 4.0, // Channel 1 (G)
                    5.0, 6.0, // Channel 2 (B)
                ],
                [1, 3, 1, 2],
            ),
            &device,
        );

        let output = gray.execute(input);

        // All channels should be equal after grayscale
        let output_data = output.to_data();
        let values: Vec<f32> = output_data.as_slice::<f32>().unwrap().to_vec();

        // Check that R, G, B channels have the same values
        // For position 0: values[0]=R, values[2]=G, values[4]=B
        let r_val0 = values[0];
        let g_val0 = values[2];
        let b_val0 = values[4];

        assert!(
            (r_val0 - g_val0).abs() < 1e-6,
            "R and G channels should be equal at position 0"
        );
        assert!(
            (g_val0 - b_val0).abs() < 1e-6,
            "G and B channels should be equal at position 0"
        );

        // For position 1: values[1]=R, values[3]=G, values[5]=B
        let r_val1 = values[1];
        let g_val1 = values[3];
        let b_val1 = values[5];

        assert!(
            (r_val1 - g_val1).abs() < 1e-6,
            "R and G channels should be equal at position 1"
        );
        assert!(
            (g_val1 - b_val1).abs() < 1e-6,
            "G and B channels should be equal at position 1"
        );
    }

    #[test]
    fn test_random_grayscale_probability_zero_never_grayscales() {
        let device = Device::default();
        let gray = RandomGrayscale::new(0.0);

        let input = Tensor::<4>::from_data(
            TensorData::new(
                vec![
                    1.0f32, 2.0, // Channel 0
                    3.0, 4.0, // Channel 1
                    5.0, 6.0, // Channel 2
                ],
                [1, 3, 1, 2],
            ),
            &device,
        );

        let output = gray.execute(input.clone());

        // With p=0.0, output should equal input
        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }
}
