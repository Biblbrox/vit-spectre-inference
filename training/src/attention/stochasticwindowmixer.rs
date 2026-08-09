
use burn::{
    module::{Module, Param},
    prelude::{Device, Tensor},
    backend::Backend,
    
tensor::{Distribution, Int, activation::softmax},
};

use crate::attention::{
    NormalizationMode, StochasticMul, StochasticSelect, TrainingMode, sinkhorn,
};

/// Stochastic window attention implementation.
/// By default, it works as a basic window attention. To apply double-stochastic behaviour,
/// you need to set StochasticSelect/StochasticMul options.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StochasticAttentionWindowConfig {
    pub embed_dim: usize,
    pub seq_length: usize,
    pub nhead: usize,
    pub kernel_size: usize,
    pub temperature: f32,
    #[serde(default)]
    pub stoch_mode: StochasticSelect,
    // This parameter describes whether matrices should be normalized by rows only
    // (stochastic) or rows and columns (double stochastic).
    #[serde(default)]
    pub norm_mode: NormalizationMode,
    #[serde(default)]
    pub score_mode: StochasticMul,
    #[serde(default)]
    pub training_mode: TrainingMode,
}

impl StochasticAttentionWindowConfig {
    pub fn new(
        embed_dim: usize,
        seq_length: usize,
        nhead: usize,
        kernel_size: usize,
        temperature: f32,
    ) -> Self {
        Self {
            embed_dim,
            seq_length,
            nhead,
            kernel_size,
            temperature,
            stoch_mode: StochasticSelect::Q,
            norm_mode: NormalizationMode::Single,
            score_mode: StochasticMul::Softmax,
            training_mode: TrainingMode::Hard,
        }
    }
}

#[derive(Module, Debug)]
pub struct StochasticAttentionWindow {
    q_mat: Param<Tensor<4>>,
    k_mat: Param<Tensor<4>>,
    v_mat: Param<Tensor<4>>,
    inv_scale: f32,
    band_bias: Param<Tensor<5>>, // [H, N, 2w+1]
    temperature: f32,
    half_width: usize,
    nhead: usize,
    dk: usize,
    window_indices: Tensor<1, Int>, // [N * bw]
    stoch_mode: StochasticSelect,
    norm_mode: NormalizationMode,
    score_mode: StochasticMul,
    train_mode: TrainingMode,
    seq_length: usize,
    
}

impl StochasticAttentionWindow {
    fn local_window(&self, x: Tensor<4>) -> Tensor<5> {
        let [b, n, h, dk] = x.dims();
        let bw = 2 * self.half_width + 1;

        let flat_idx = self.window_indices.clone();

        // [B, N, H, dk] → [B, N*bw, H, dk]
        let gathered = x.select(1, flat_idx);

        // Restore window structure and move bw to the last dim
        gathered
            .reshape([b, n, bw, h, dk]) // [B, N, bw, H, dk]
            .permute([0, 1, 3, 4, 2]) // [B, N, H, dk, bw]
    }

    fn calc_qkv_soft(&self) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let (do_q, do_k, do_v) = self.stoch_mode.flags();
        let t = self.temperature;

        let apply = |do_it: bool, mat: Tensor<4>| -> Tensor<4> {
            if do_it {
                sinkhorn(mat, t, self.norm_mode)
            } else {
                mat
            }
        };

        (
            apply(do_q, self.q_mat.val()),
            apply(do_k, self.k_mat.val()),
            apply(do_v, self.v_mat.val()),
        )
    }

    fn calc_qkv_hard(&self, x: Tensor<4>) -> (Tensor<5>, Tensor<5>, Tensor<5>) {
        let dk = x.dims()[3];
        let (do_q, do_k, do_v) = self.stoch_mode.flags();

        // Either select the argmax "hard" index along dk, or apply the matrix as a linear map.
        let apply = |do_it: bool, mat: Tensor<4>| -> Tensor<4> {
            if do_it {
                x.clone().select(3, mat.argmax(2).reshape([dk]))
            } else {
                x.clone().matmul(mat)
            }
        };

        let q = apply(do_q, self.q_mat.val()).unsqueeze_dim(3);
        let k = self.local_window(apply(do_k, self.k_mat.val()));
        let v = self.local_window(apply(do_v, self.v_mat.val()));

        (q, k, v)
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self.train_mode {
            TrainingMode::Soft => self.forward_soft(x),
            TrainingMode::Hard | TrainingMode::Mixed => self.forward_hard(x),
        }
    }

    fn calc_scores(&self, q: Tensor<5>, k_win: Tensor<5>) -> Tensor<5> {
        let scores = q.matmul(k_win) * self.inv_scale + self.band_bias.val();
        match self.score_mode {
            StochasticMul::Softmax => softmax(scores, 4),
            StochasticMul::Sinkhorn => sinkhorn(scores, self.temperature, self.norm_mode),
            StochasticMul::None => scores,
        } // [B,N,H,1,bw]
    }

    pub fn forward_hard(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.nhead;

        let x = x.reshape([b, n, h, dk]);

        let (q, k, v) = self.calc_qkv_hard(x);
        let p = self.calc_scores(q, k);
        let out = v.matmul(p.transpose());

        out.reshape([b, n, e])
    }

    pub fn forward_soft(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.nhead;

        let x = x.reshape([b, n, h, dk]);

        let (w_q, w_k, w_v) = self.calc_qkv_soft();
        let q = x.clone().matmul(w_q).unsqueeze_dim(3); // [B,N,H,1,dk]
        let k = x.clone().matmul(w_k); // [B,N,H,dk]
        let v = x.matmul(w_v); // [B,N,H,dk]

        let k = self.local_window(k); // [B,N,H,dk,bw]
        let v = self.local_window(v); // [B,N,H,dk,bw]
        let p = self.calc_scores(q, k); // [B,N,H,1,bw]

        let out = v.matmul(p.transpose()); // [B,N,H,dk,1]

        out.reshape([b, n, e])
    }
}

impl StochasticAttentionWindowConfig {
    pub fn init(&self, device: &Device) -> StochasticAttentionWindow {
        let w = (self.kernel_size - 1) / 2;
        let window = 2 * w + 1;
        let dk = self.embed_dim / self.nhead; // head dim
        let n = self.seq_length;

        let logit_std = (1.0 / dk as f64).sqrt();

        let pos = Tensor::<1, Int>::arange(0..n as i64, device).reshape([1, n, 1, 1]);
        let offsets = Tensor::<1, Int>::arange(-(w as i64)..(w as i64 + 1), device)
            .reshape([1, 1, 1, window]);
        let window_indices = (pos + offsets).clamp(0, n as i64 - 1); // [1, N, 1, bw]

        let init_logits = || {
            Param::from_tensor(Tensor::<4>::random(
                // [1, self.seq_length, dk, dk],
                [1, 1, dk, dk],
                Distribution::Normal(0.0, logit_std),
                device,
            ))
            .set_require_grad(true)
        };

        StochasticAttentionWindow {
            band_bias: Param::from_tensor(Tensor::<5>::zeros(
                [1, self.seq_length, self.nhead, 1, window],
                device,
            ))
            .set_require_grad(true),
            temperature: self.temperature,
            half_width: w,
            nhead: self.nhead,
            q_mat: init_logits(),
            k_mat: init_logits(),
            v_mat: init_logits(),
            dk,
            inv_scale: 1.0 / (dk as f32).sqrt(),
            window_indices: window_indices.clone().reshape([n * window]),
            stoch_mode: self.stoch_mode,
            seq_length: self.seq_length,
            norm_mode: self.norm_mode,
            score_mode: self.score_mode,
            train_mode: self.training_mode,
        }
    }
}

/// Since we use select instead of unfold as a hack to make window creation faster,
/// we must be sure that it still produces the same results as unfold
#[cfg(test)]
mod tests {
    use burn::{
        backend::{Flex, flex::FlexDevice},
        
tensor::{Int, Shape, Tensor, TensorData, ops::PadMode, s},
    };

    type B = Flex;
    type TestDevice = FlexDevice;

    fn device() -> TestDevice {
        Default::default()
    }

    /// Build window_indices (flat [N*bw]) the same way init() does.
    fn make_window_indices(n: usize, w: usize, device: &TestDevice) -> Tensor<1, Int> {
        let window = 2 * w + 1;
        let pos = Tensor::<1, Int>::arange(0..n as i64, device).reshape([1, n, 1, 1]);
        let offsets = Tensor::<1, Int>::arange(-(w as i64)..(w as i64 + 1), device)
            .reshape([1, 1, 1, window]);
        (pos + offsets).clamp(0, n as i64 - 1).reshape([n * window])
    }

    /// select-based local_window: drop-in replacement for pad+unfold.
    /// x: [B, N, H, dk] → [B, N, H, dk, bw]
    fn local_window_select(
        x: Tensor<4>,
        window_indices: Tensor<1, Int>,
        w: usize,
    ) -> Tensor<5> {
        let [b, n, h, dk] = x.dims();
        let bw = 2 * w + 1;
        let gathered = x.select(1, window_indices); // [B, N*bw, H, dk]
        gathered
            .reshape([b, n, bw, h, dk])
            .swap_dims(2, 3) // [B, N, H, bw, dk]
            .transpose() // [B, N, H, dk, bw]
    }

    /// Original pad+unfold approach.
    /// x: [B, N, H, dk] → [B, N, H, dk, bw]
    fn local_window_unfold(x: Tensor<4>, w: usize) -> Tensor<5> {
        let bw = 2 * w + 1;
        let x_pad = x.pad([(0, 0), (w, w), (0, 0), (0, 0)], PadMode::Constant(0.0));
        x_pad.unfold(1, bw, 1) // [B, N, H, dk, bw]
    }

    fn max_abs_diff(a: Tensor<5>, b: Tensor<5>) -> f32 {
        (a - b).abs().max().into_scalar()
    }

    /// Test 1: Interior tokens — both approaches must agree exactly.
    #[test]
    fn test_interior_tokens_match() {
        let device = device();
        let (b, n, h, dk, w) = (2, 8, 2, 4, 1); // w=1 → bw=3, interior = tokens 1..7
        let bw = 2 * w + 1;

        // Deterministic input: value = position index so we can reason about it
        let data: Vec<f32> = (0..(b * n * h * dk) as i32).map(|i| i as f32).collect();
        let x =
            Tensor::<4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w);
        let unfold_out = local_window_unfold(x, w);

        assert_eq!(select_out.dims(), [b, n, h, dk, bw]);
        assert_eq!(unfold_out.dims(), [b, n, h, dk, bw]);

        // Slice out interior tokens only (skip first and last w tokens)
        let select_interior = select_out.slice_dim(1, s![w..n - w]);
        let unfold_interior = unfold_out.slice_dim(1, s![w..n - w]);

        let diff = max_abs_diff(select_interior, unfold_interior);
        assert!(diff < 1e-5, "Interior tokens differ: max abs diff = {diff}");
    }

    /// Test 2: Boundary behaviour difference is expected and documented.
    #[test]
    fn test_boundary_behaviour_difference() {
        let device = device();
        let (b, n, h, dk, w) = (1, 6, 1, 2, 2); // w=2 → first/last 2 tokens are boundary

        let data: Vec<f32> = (0..(b * n * h * dk) as i32)
            .map(|i| i as f32 + 1.0)
            .collect();
        let x =
            Tensor::<4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w);
        let unfold_out = local_window_unfold(x, w);

        // Interior tokens [w..n-w] must match
        let select_interior = select_out.clone().slice_dim(1, s![w..n - w]);
        let unfold_interior = unfold_out.clone().slice_dim(1, s![w..n - w]);
        let interior_diff = max_abs_diff(select_interior, unfold_interior);
        assert!(
            interior_diff < 1e-5,
            "Interior tokens should match, diff = {interior_diff}"
        );

        // Boundary tokens must differ (clamp != zero padding, so diff > 0)
        let select_boundary = select_out.slice_dim(1, s![0..w]);
        let unfold_boundary = unfold_out.slice_dim(1, s![0..w]);
        let boundary_diff = max_abs_diff(select_boundary, unfold_boundary);
        assert!(
            boundary_diff > 0.0,
            "Boundary tokens should differ between clamp and zero-pad"
        );
    }

    /// Random input, large window — interior agreement holds at scale.
    #[test]
    fn test_random_input_interior_agreement() {
        let device = device();
        let (b, n, h, dk, w) = (3, 16, 4, 8, 3);

        // Use a fixed seed-like pattern for reproducibility without a seeded RNG
        let data: Vec<f32> = (0..(b * n * h * dk) as i32)
            .map(|i| ((i as f32 * 1.6180339) % 7.0) - 3.5) // pseudo-random spread
            .collect();
        let x =
            Tensor::<4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w);
        let unfold_out = local_window_unfold(x, w);

        let select_interior = select_out.slice_dim(1, s![w..n - w]);
        let unfold_interior = unfold_out.slice_dim(1, s![w..n - w]);

        let diff = max_abs_diff(select_interior, unfold_interior);
        assert!(diff < 1e-5, "Random input interior diff = {diff}");
    }

    /// Window of 1 (w=0, bw=1) — trivially each token attends only itself.
    /// Both approaches must return the input unchanged (modulo reshape).
    #[test]
    fn test_window_size_one() {
        let device = device();
        let (b, n, h, dk, w) = (2, 5, 2, 4, 0);

        let data: Vec<f32> = (0..(b * n * h * dk) as i32).map(|i| i as f32).collect();
        let x =
            Tensor::<4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w); // [B,N,H,dk,1]
        let unfold_out = local_window_unfold(x.clone(), w); // [B,N,H,dk,1]

        // Both should equal x reshaped to [B,N,H,dk,1]
        let x_expanded = x.unsqueeze_dim(4);
        let diff_select = max_abs_diff(select_out, x_expanded.clone());
        let diff_unfold = max_abs_diff(unfold_out, x_expanded);

        assert!(
            diff_select < 1e-5,
            "w=0 select should equal input, diff={diff_select}"
        );
        assert!(
            diff_unfold < 1e-5,
            "w=0 unfold should equal input, diff={diff_unfold}"
        );
    }

    /// Output shape is correct for various configurations.
    #[test]
    fn test_output_shapes() {
        let device = device();

        for (b, n, h, dk, w) in [(1, 4, 1, 2, 1), (2, 8, 4, 16, 2), (1, 16, 8, 32, 4)] {
            let bw = 2 * w + 1;
            let data = vec![0.0f32; b * n * h * dk];
            let x = Tensor::<4>::from_data(
                TensorData::new(data, Shape::new([b, n, h, dk])),
                &device,
            );

            let win_idx = make_window_indices(n, w, &device);
            let select_out = local_window_select(x.clone(), win_idx, w);
            let unfold_out = local_window_unfold(x, w);

            assert_eq!(
                select_out.dims(),
                [b, n, h, dk, bw],
                "select shape wrong for b={b} n={n} h={h} dk={dk} w={w}"
            );
            assert_eq!(
                unfold_out.dims(),
                [b, n, h, dk, bw],
                "unfold shape wrong for b={b} n={n} h={h} dk={dk} w={w}"
            );
        }
    }

    /// Specific values — manually verify a tiny case end-to-end.
    /// x = [[1,2],[3,4],[5,6]] (n=3, dk=2, b=1, h=1, w=1)
    /// For token n=1 (middle), window should be [token0, token1, token2]
    /// i.e. along dk dim: [[1,2],[3,4],[5,6]] → last dim of output at n=1
    #[test]
    fn test_specific_values() {
        let device = device();
        let (b, n, h, dk, w) = (1, 3, 1, 2, 1);
        let bw = 2 * w + 1;

        // x[0, 0, 0, :] = [1, 2]
        // x[0, 1, 0, :] = [3, 4]
        // x[0, 2, 0, :] = [5, 6]
        let x = Tensor::<4>::from_data(
            TensorData::new(vec![1.0f32, 2., 3., 4., 5., 6.], Shape::new([b, n, h, dk])),
            &device,
        );

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w); // [1,3,1,2,3]
        let unfold_out = local_window_unfold(x, w);

        // At n=1 (middle token), both approaches see tokens [0,1,2] — no boundary effect
        // select_out[0, 1, 0, :, :] should be [[1,3,5],[2,4,6]]
        // i.e. dk=0 row: [tok0_dk0, tok1_dk0, tok2_dk0] = [1, 3, 5]
        //      dk=1 row: [tok0_dk1, tok1_dk1, tok2_dk1] = [2, 4, 6]
        let select_mid = select_out.slice_dim(1, s![1..2]).reshape([dk, bw]);
        let unfold_mid = unfold_out.slice_dim(1, s![1..2]).reshape([dk, bw]);

        let expected = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 3., 5., 2., 4., 6.], Shape::new([dk, bw])),
            &device,
        );

        let diff_select = (select_mid - expected.clone()).abs().max().into_scalar();
        let diff_unfold = (unfold_mid - expected).abs().max().into_scalar();

        assert!(
            diff_select < 1e-5,
            "select middle token wrong: diff={diff_select}"
        );
        assert!(
            diff_unfold < 1e-5,
            "unfold middle token wrong: diff={diff_unfold}"
        );
    }
}
