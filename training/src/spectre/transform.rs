use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, SQRT_2, TAU};

use burn::serde::{Deserialize, Serialize};

use burn::tensor::Tensor;
use burn::tensor::Device;

/// Normalised DCT-II matrix [n, n].
/// Row k is the k-th orthonormal DCT-II basis vector.
fn dct_matrix(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for k in 0..n {
        let scale = if k == 0 {
            (1.0 / n as f32).sqrt()
        } else {
            (2.0 / n as f32).sqrt()
        };
        for i in 0..n {
            m[k * n + i] = scale * (PI * k as f32 * (2 * i + 1) as f32 / (2.0 * n as f32)).cos();
        }
    }
    m
}

/// Build the rectangular projection matrix of shape [out_features, in_features].
///
/// Strategy:
/// - Compute the DCT-II basis for max(in, out) dimensions.
/// - Each output row k selects basis vector `k % in_features` of the
///   DCT computed over `in_features`, then truncates / zero-pads to
///   `in_features` columns.
/// - When out_features <= in_features  → first `out_features` DCT rows
///   (low-to-high frequency, no repetition).
/// - When out_features >  in_features  → frequencies cycle, giving
///   the decoder a full-rank initialisation without an assert.
pub fn build_dct_projection(in_features: usize, out_features: usize) -> Vec<f32> {
    // DCT basis is always square over the input dimension
    let basis = dct_matrix(in_features);
    let mut w = vec![0.0f32; out_features * in_features];

    for k in 0..out_features {
        let src_row = k % in_features; // cycles when out > in
        for i in 0..in_features {
            w[k * in_features + i] = basis[src_row * in_features + i];
        }
    }
    w
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SpectralTransform {
    Hadamard,
    Hartley,
    Cosine,
    Sine,
    None,
}

impl SpectralTransform {
    pub fn xform(&self, order: usize, device: &Device) -> Tensor<2> {
        match self {
            SpectralTransform::Hadamard => Self::hadamard(order, device),
            SpectralTransform::Hartley => Self::hartley(order, device),
            SpectralTransform::Cosine => Self::cosine(order, device),
            SpectralTransform::Sine => Self::sine(order, device),
            SpectralTransform::None => Tensor::eye(1 << order, device),
        }
    }

    fn hadamard(order: usize, device: &Device) -> Tensor<2> {
        if order == 0 {
            return Tensor::ones([1, 1], device);
        }

        let mut x = Tensor::from_floats([[1.0, 1.0], [1.0, -1.0]], device);

        for r in 1..order {
            let neg = x.clone().neg();
            x = x.repeat(&[2, 2]);

            let len = 1 << r;
            x = x.slice_assign([len.., len..], neg);
        }

        x.mul_scalar(FRAC_1_SQRT_2.powi(order as i32))
    }

    fn hartley(order: usize, device: &Device) -> Tensor<2> {
        if order == 0 {
            return Tensor::ones([1, 1], device);
        }

        let n: usize = 1 << order;
        let c = TAU / n as f32;

        let x = Tensor::ones([n - 1, n - 1], device);
        let x = x.cumsum(0);
        let x = x.cumsum(1);
        let x = x.pad((1, 0, 1, 0), 0);
        let x = x.mul_scalar(c);
        let cos = x.clone().cos();
        let sin = x.sin();
        let x = cos + sin;

        x.div_scalar(n.isqrt() as f32)
    }

    fn cosine(order: usize, device: &Device) -> Tensor<2> {
        if order == 0 {
            return Tensor::ones([1, 1], device);
        }

        let n: usize = 1 << order;
        let mut container = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let val = i as f32 * (2.0 * j as f32 + 1.0) * FRAC_PI_2 / n as f32;
                match i {
                    0 => container.push(val.cos()),
                    _ => container.push(val.cos() * SQRT_2),
                }
            }
        }

        let x = Tensor::<1>::from_floats(container.as_slice(), device);
        let x = x.reshape([n, n]);

        x.div_scalar(n.isqrt() as f32)
    }

    fn sine(order: usize, device: &Device) -> Tensor<2> {
        if order == 0 {
            return Tensor::ones([1, 1], device);
        }

        let n: usize = 1 << order;
        let c = (n + 1) as f32;

        let x = Tensor::ones([n, n], device);
        let x = x.cumsum(0);
        let x = x.cumsum(1);
        let x = x.mul_scalar(PI / c);
        let x = x.sin();

        x.mul_scalar(SQRT_2 / c.sqrt())
    }
}
