use burn::{
    Tensor,
    tensor::{TensorPrimitive, backend::Backend, ops::FloatTensor},
};

pub mod cascadedattention;
pub mod csp_attention;
pub mod dsthaattention;
pub mod learnedmixer;
pub mod self_attention;
pub mod sinkformer;
pub mod staticmixer;
pub mod stochasticmixer;
pub mod stochasticmixerstatic;
pub mod stochasticmixerstaticwindow;
pub mod stochasticwindowmixer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum NormalizationMode {
    Single,
    #[default]
    Double,
}

pub fn sinkhorn<B: Backend, const D: usize>(
    s: Tensor<B, D>,
    temp: f32,
    mode: NormalizationMode,
) -> Tensor<B, D> {
    assert!(D >= 2, "sinkhorn requires at least 2 dimensions");
    let (last, second_last) = (D - 1, D - 2);
    Tensor::from_primitive(TensorPrimitive::Float(match mode {
        NormalizationMode::Double => {
            sinkhorn_double::<B>(s.into_primitive().tensor(), temp, last, second_last)
        }
        NormalizationMode::Single => {
            sinkhorn_single::<B>(s.into_primitive().tensor(), temp, second_last)
        }
    }))
}

fn calc_qkv_soft<B: Backend>(
    temperature: f32,
    stoch_mode: StochasticSelect,
    norm_mode: NormalizationMode,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let (do_q, do_k, do_v) = stoch_mode.flags();
    let t = temperature;

    let apply = |do_it: bool, mat: Tensor<B, 4>| -> Tensor<B, 4> {
        if do_it {
            sinkhorn(mat, t, norm_mode)
        } else {
            mat
        }
    };

    (apply(do_q, q), apply(do_k, k), apply(do_v, v))
}

fn calc_qkv_hard<B: Backend>(
    x: Tensor<B, 4>,
    stoch_mode: StochasticSelect,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let dk = x.dims()[3];
    let (do_q, do_k, do_v) = stoch_mode.flags();

    // Either select the argmax "hard" index along dk, or apply the matrix as a linear map.
    let apply = |do_it: bool, mat: Tensor<B, 4>| -> Tensor<B, 4> {
        if do_it {
            x.clone().select(3, mat.argmax(2).reshape([dk]))
        } else {
            x.clone().matmul(mat)
        }
    };

    (apply(do_q, q), apply(do_k, k), apply(do_v, v))
}

pub fn sinkhorn_iter<B: Backend, const D: usize>(
    s: Tensor<B, D>,
    temp: f32,
    iters: usize,
    mode: NormalizationMode,
) -> Tensor<B, D> {
    assert!(D >= 2, "sinkhorn requires at least 2 dimensions");
    let (last, second_last) = (D - 1, D - 2);
    let mut prim = s.into_primitive().tensor();
    for _i in 0..iters {
        prim = match mode {
            NormalizationMode::Double => sinkhorn_double::<B>(prim, temp, last, second_last),
            NormalizationMode::Single => sinkhorn_single::<B>(prim, temp, last),
        }
    }
    Tensor::from_primitive(TensorPrimitive::Float(prim))
}

/// produces a doubly-stochastic matrix over (second_last, last)
fn sinkhorn_double<B: Backend>(
    tensor: FloatTensor<B>,
    temp: f32,
    last: usize,
    second_last: usize,
) -> FloatTensor<B> {
    let tensor = B::float_div_scalar(tensor, burn::tensor::Scalar::Float(temp as f64));
    let max = B::float_max_dim(B::float_detach(tensor.clone()), last);
    let shifted = B::float_sub(tensor, max);
    let exp = B::float_exp(shifted);
    let sum = B::float_sum_dim(exp.clone(), last);
    let tensor = B::float_div(exp, sum);

    let max = B::float_max_dim(B::float_detach(tensor.clone()), second_last);
    let shifted = B::float_sub(tensor, max);
    let exp = B::float_exp(shifted);
    let sum = B::float_sum_dim(exp.clone(), second_last);
    B::float_div(exp, sum)
}

/// produces a row-stochastic matrix over `last`
fn sinkhorn_single<B: Backend>(tensor: FloatTensor<B>, temp: f32, last: usize) -> FloatTensor<B> {
    let tensor = B::float_div_scalar(tensor, burn::tensor::Scalar::Float(temp as f64));
    let max = B::float_max_dim(B::float_detach(tensor.clone()), last);
    let shifted = B::float_sub(tensor, max);
    let exp = B::float_exp(shifted);
    let sum = B::float_sum_dim(exp.clone(), last);
    B::float_div(exp, sum)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum StochasticSelect {
    Q,
    K,
    #[default]
    Qk,
    Qkv,
    None,
}

impl StochasticSelect {
    /// (apply to q, apply to k, apply to v)
    fn flags(self) -> (bool, bool, bool) {
        match self {
            StochasticSelect::Q => (true, false, false),
            StochasticSelect::K => (false, true, false),
            StochasticSelect::Qk => (true, true, false),
            StochasticSelect::Qkv => (true, true, true),
            StochasticSelect::None => (false, false, false),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum StochasticMul {
    Sinkhorn,
    #[default]
    Softmax,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum TrainingMode {
    Soft,
    Mixed,
    #[default]
    Hard,
}

#[cfg(test)]
mod tests {
    use burn::{Tensor, backend::Flex, tensor::TensorData};

    use crate::attention::NormalizationMode;

    type B = Flex;
    type Device = burn::backend::flex::FlexDevice;

    fn naive_sinkhorn_double_2d(matrix: &[f32], rows: usize, cols: usize, temp: f32) -> Vec<f32> {
        let mut mat = matrix.to_vec();

        // divide by temperature
        for v in mat.iter_mut() {
            *v /= temp;
        }

        // row normalization (dim 3): for each row, subtract max, exp, normalize
        for i in 0..rows {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..cols {
                let v = f64::from(mat[i * cols + j]);
                if v > max_val {
                    max_val = v;
                }
            }
            let mut row_sum = 0.0f64;
            for j in 0..cols {
                mat[i * cols + j] = ((f64::from(mat[i * cols + j]) - max_val).exp()) as f32;
                row_sum += f64::from(mat[i * cols + j]);
            }
            if row_sum > 0.0 {
                for j in 0..cols {
                    mat[i * cols + j] /= row_sum as f32;
                }
            }
        }

        // column normalization (dim 2): for each col, subtract max, exp, normalize
        for j in 0..cols {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..rows {
                let v = f64::from(mat[i * cols + j]);
                if v > max_val {
                    max_val = v;
                }
            }
            let mut col_sum = 0.0f64;
            for i in 0..rows {
                mat[i * cols + j] = ((f64::from(mat[i * cols + j]) - max_val).exp()) as f32;
                col_sum += f64::from(mat[i * cols + j]);
            }
            if col_sum > 0.0 {
                for i in 0..rows {
                    mat[i * cols + j] /= col_sum as f32;
                }
            }
        }

        mat
    }

    /// Full iterative Sinkhorn (matches sinkhorn_iter).
    fn naive_sinkhorn_iter_2d(
        matrix: &[f32],
        rows: usize,
        cols: usize,
        _temp: f32,
        iters: usize,
    ) -> Vec<f32> {
        let mut mat = matrix.to_vec();
        for _iter in 0..iters {
            // row normalization (dim 3)
            for i in 0..rows {
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..cols {
                    let v = f64::from(mat[i * cols + j]);
                    if v > max_val {
                        max_val = v;
                    }
                }
                let mut row_sum = 0.0f64;
                for j in 0..cols {
                    mat[i * cols + j] = ((f64::from(mat[i * cols + j]) - max_val).exp()) as f32;
                    row_sum += f64::from(mat[i * cols + j]);
                }
                if row_sum > 0.0 {
                    for j in 0..cols {
                        mat[i * cols + j] /= row_sum as f32;
                    }
                }
            }
            // column normalization (dim 2)
            for j in 0..cols {
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..rows {
                    let v = f64::from(mat[i * cols + j]);
                    if v > max_val {
                        max_val = v;
                    }
                }
                let mut col_sum = 0.0f64;
                for i in 0..rows {
                    mat[i * cols + j] = ((f64::from(mat[i * cols + j]) - max_val).exp()) as f32;
                    col_sum += f64::from(mat[i * cols + j]);
                }
                if col_sum > 0.0 {
                    for i in 0..rows {
                        mat[i * cols + j] /= col_sum as f32;
                    }
                }
            }
        }
        mat
    }

    fn make_tensor_4d(data: &[f32], shape: &[usize]) -> Tensor<B, 4> {
        let device = Device::default();
        Tensor::<B, 4>::from_data(TensorData::new(data.to_vec(), shape.to_vec()), &device)
    }

    /// Flat index for tensor layout [batch, heads, rows, cols]
    fn flat_idx(
        b: usize,
        h: usize,
        r: usize,
        c: usize,
        heads: usize,
        rows: usize,
        cols: usize,
    ) -> usize {
        b * heads * rows * cols + h * rows * cols + r * cols + c
    }

    /// Extract tensor data as owned Vec<f32> (avoids lifetime issues with .to_data())
    fn to_vec(output: &Tensor<B, 4>) -> Vec<f32> {
        output.to_data().as_slice::<f32>().unwrap().to_vec()
    }

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// Assert that a tensor is doubly stochastic: every row and column sums to 1.0.
    fn assert_doubly_stochastic(d: &[f32], rows: usize, cols: usize, tol: f32) {
        for row in 0..rows {
            let s: f32 = (0..cols).map(|c| d[row * cols + c]).sum();
            assert!((s - 1.0).abs() < tol, "row {} sum={} (tol={})", row, s, tol);
        }
        for col in 0..cols {
            let s: f32 = (0..rows).map(|r| d[r * cols + col]).sum();
            assert!((s - 1.0).abs() < tol, "col {} sum={} (tol={})", col, s, tol);
        }
    }

    /// Assert that two flat vectors match element-wise within tolerance.
    fn assert_elementwise_eq(actual: &[f32], expected: &[f32], tol: f32) {
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "idx {}: {} vs {} (tol={})",
                i,
                a,
                e,
                tol
            );
        }
    }

    /// Assert that high-contrast input produces a more peaked distribution than low-contrast.
    fn assert_more_peaked(high: &Tensor<B, 4>, low: &Tensor<B, 4>) {
        let h = to_vec(high);
        let l = to_vec(low);
        let max_h: f32 = h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_l: f32 = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_h >= max_l,
            "high-contrast should be more peaked: {} vs {}",
            max_h,
            max_l
        );
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    #[test]
    fn test_sinkhorn_double_stochastic() {
        let device = Device::default();
        let input = Tensor::<B, 4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        let output = super::sinkhorn(input, 1.0, NormalizationMode::Double);
        let d = to_vec(&output);

        assert_doubly_stochastic(&d, 2, 2, 1e-5);

        // non-negative
        for v in &d {
            assert!(*v >= -1e-6, "negative={}", v);
        }
    }

    #[test]
    fn test_sinkhorn_double_vs_naive_2d() {
        let n_rows = 4;
        let n_cols = 5;
        let data: Vec<f32> = (0..n_rows * n_cols)
            .map(|i| (i % 20) as f32 + 1.0)
            .collect();

        let expected = naive_sinkhorn_double_2d(&data, n_rows, n_cols, 0.5);
        let input = make_tensor_4d(&data, &[1, 1, n_rows, n_cols]);
        let output = super::sinkhorn(input, 0.5, NormalizationMode::Double);
        let out = to_vec(&output);

        assert_elementwise_eq(&out, &expected, 1e-5);
    }

    #[test]
    fn test_sinkhorn_iter_convergence() {
        let n_rows = 4;
        let n_cols = 4;
        let data: Vec<f32> = (0..n_rows * n_cols).map(|i| (i % 16 + 1) as f32).collect();

        let input = make_tensor_4d(&data, &[1, 1, n_rows, n_cols]);
        let out_5 = super::sinkhorn_iter(input.clone(), 1.0, 5, NormalizationMode::Double);
        let out_10 = super::sinkhorn_iter(input.clone(), 1.0, 10, NormalizationMode::Double);
        let out_20 = super::sinkhorn_iter(input, 1.0, 20, NormalizationMode::Double);

        // compare each iteration count vs naive
        for (iters, actual_vec) in [(5, to_vec(&out_5)), (10, to_vec(&out_10))] {
            let expected = naive_sinkhorn_iter_2d(&data, n_rows, n_cols, 1.0, iters);
            assert_elementwise_eq(&actual_vec, &expected, 1e-4);
        }

        // 20 iters should be very close to 10
        let s10 = to_vec(&out_10);
        let s20 = to_vec(&out_20);
        let max_diff: f32 = s10
            .iter()
            .zip(s20.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 1e-6, "convergence gap={}", max_diff);
    }

    #[test]
    fn test_sinkhorn_temperature_scaling() {
        let data: Vec<f32> = vec![0.1, 10.0, 10.0, 0.1];
        let input = make_tensor_4d(&data, &[1, 1, 2, 2]);

        let high = super::sinkhorn(input.clone(), 10.0, NormalizationMode::Double);
        let low = super::sinkhorn(input, 0.1, NormalizationMode::Double);

        let h = to_vec(&high);
        let l = to_vec(&low);

        let mean_h: f32 = h.iter().sum::<f32>() / h.len() as f32;
        let var_h: f32 = h.iter().map(|x| (x - mean_h).powi(2)).sum::<f32>() / h.len() as f32;

        let mean_l: f32 = l.iter().sum::<f32>() / l.len() as f32;
        let var_l: f32 = l.iter().map(|x| (x - mean_l).powi(2)).sum::<f32>() / l.len() as f32;

        assert!(var_h < var_l, "high-var={} low-var={}", var_h, var_l);
    }

    #[test]
    fn test_sinkhorn_preserves_shape() {
        let shapes = vec![
            ([1, 1, 2, 2], "small"),
            ([2, 3, 4, 5], "medium batch"),
            ([1, 8, 10, 10], "multi-head attention"),
        ];
        for (shape, name) in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f32> = (0..total).map(|i| (i % 7 + 1) as f32).collect();
            let input = make_tensor_4d(&data, &shape);
            let output = super::sinkhorn(input, 1.0, NormalizationMode::Double);
            assert_eq!(output.shape().dims(), shape, "shape mismatch for {}", name);
        }
    }

    #[test]
    fn test_sinkhorn_batch_heads_independent() {
        // Two independent matrices: batch=1, heads=2
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // head 0
            5.0, 6.0, 7.0, 8.0, // head 1
        ];
        let input = make_tensor_4d(&data, &[1, 2, 2, 2]);
        let output = super::sinkhorn(input, 1.0, NormalizationMode::Double);
        let d = to_vec(&output);

        for head in 0..2 {
            let base = flat_idx(0, head, 0, 0, 2, 2, 2);
            let head_slice = &d[base..base + 4];
            assert_doubly_stochastic(head_slice, 2, 2, 1e-5);
        }
    }

    #[test]
    fn test_sinkhorn_high_contrast_input() {
        // High-contrast input should produce more peaked distribution
        let device = Device::default();
        let high_contrast = Tensor::<B, 4>::from_floats([[[[10.0, 0.1], [0.1, 10.0]]]], &device);
        let low_contrast = Tensor::<B, 4>::from_floats([[[[1.0, 0.9], [0.9, 1.0]]]], &device);

        assert_more_peaked(&high_contrast, &low_contrast);
    }

    #[test]
    fn test_sinkhorn_all_positive_values() {
        for size in [2, 3, 4, 5, 8] {
            let data: Vec<f32> = (0..size * size)
                .map(|i| ((i % 10 + 1) as f32).powf(0.5))
                .collect();
            let input = make_tensor_4d(&data, &[1, 1, size, size]);
            let output = super::sinkhorn(input, 1.0, NormalizationMode::Double);
            for &v in to_vec(&output).iter() {
                assert!(v >= -1e-7, "{}x{} negative={}", size, size, v);
            }
        }
    }

    #[test]
    fn test_sinkhorn_iter_matches_multiple_naive_passes() {
        let n_rows = 3;
        let n_cols = 4;
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let input = make_tensor_4d(&data, &[1, 1, n_rows, n_cols]);
        let output = super::sinkhorn_iter(input, 0.7, 8, NormalizationMode::Double);
        let expected = naive_sinkhorn_iter_2d(&data, n_rows, n_cols, 0.7, 8);
        let out = to_vec(&output);

        assert_elementwise_eq(&out, &expected, 1e-4);
    }

    #[test]
    fn test_sinkhorn_low_temperature_peaked() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 1.0];
        let input = make_tensor_4d(&data, &[1, 1, 2, 2]);

        let output_low = super::sinkhorn(input.clone(), 0.1, NormalizationMode::Double);
        let output_high = super::sinkhorn(input, 5.0, NormalizationMode::Double);

        assert_more_peaked(&output_low, &output_high);
    }

    #[test]
    fn test_sinkhorn_double_single_consistency() {
        let data: Vec<f32> = (0..16).map(|i| (i % 16 + 1) as f32).collect();
        let input = make_tensor_4d(&data, &[1, 1, 4, 4]);
        let output = super::sinkhorn(input, 1.0, NormalizationMode::Double);
        let d = to_vec(&output);

        assert_doubly_stochastic(&d, 4, 4, 1e-5);
    }

    #[test]
    fn test_naive_reference_uniform_input() {
        let uniform: Vec<f32> = vec![1.0; 9];
        let result = naive_sinkhorn_double_2d(&uniform, 3, 3, 1.0);
        for &v in &result {
            assert!(
                (v - 1.0 / 3.0).abs() < 1e-6,
                "uniform should give uniform: got {}",
                v
            );
        }
    }

    #[test]
    fn test_sinkhorn_large_matrix() {
        let size = 10;
        let data: Vec<f32> = (0..size * size)
            .map(|i| ((i % 50 + 1) as f32).ln().max(0.1))
            .collect();
        let input = make_tensor_4d(&data, &[1, 1, size, size]);
        // Use sinkhorn_iter with multiple passes for better convergence on large matrices
        let output = super::sinkhorn_iter(input, 1.0, 20, NormalizationMode::Double);
        let d = to_vec(&output);

        assert_doubly_stochastic(&d, size, size, 1e-3);
    }

    #[test]
    fn test_sinkhorn_iter_convergence_to_double_stochastic() {
        // Use a square matrix for proper double stochasticity check
        let size = 4;
        let data: Vec<f32> = (0..size * size).map(|i| (i % 16 + 1) as f32).collect();
        let input = make_tensor_4d(&data, &[1, 1, size, size]);
        // After many iterations of sinkhorn_double, should be doubly stochastic
        let output = super::sinkhorn_iter(input, 1.0, 50, NormalizationMode::Double);
        let d = to_vec(&output);

        assert_doubly_stochastic(&d, size, size, 1e-4);
    }

    #[test]
    fn test_sinkhorn_iter_vs_naive_reference() {
        let n_rows = 6;
        let n_cols = 6;
        let single_matrix: Vec<f32> = (0..n_rows * n_cols)
            .map(|i| ((i % 36 + 1) as f32).sqrt())
            .collect();

        // Repeat the same matrix for all batch/head positions so naive and GPU match
        let num_slices = 2 * 4;
        let data: Vec<f32> = (0..num_slices)
            .flat_map(|_| single_matrix.clone())
            .collect();

        let input = make_tensor_4d(&data, &[2, 4, n_rows, n_cols]);
        let output = super::sinkhorn_iter(input, 0.3, 15, NormalizationMode::Double);
        let expected = naive_sinkhorn_iter_2d(&single_matrix, n_rows, n_cols, 0.3, 15);
        let out = to_vec(&output);

        for b in 0..2 {
            for h in 0..4 {
                let base = flat_idx(b, h, 0, 0, 4, n_rows, n_cols);
                let head_out = &out[base..base + n_rows * n_cols];
                assert_elementwise_eq(head_out, &expected, 1e-3);
            }
        }
    }
}
