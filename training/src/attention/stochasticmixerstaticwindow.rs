use burn::{
    module::{Module, Param},
    prelude::{Device, Tensor},
    tensor::{Distribution, Int, activation::softmax},
};

use crate::attention::{
    NormalizationMode, StochasticMul, StochasticSelect, TrainingMode, sinkhorn,
};

/// Stochastic window attention implementation.
/// By default, it works as a basic window attention. To apply double-stochastic behaviour,
/// you need to set StochasticSelect/StochasticMul options.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StochasticAttentionStaticWindowConfig {
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

impl StochasticAttentionStaticWindowConfig {
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
            score_mode: StochasticMul::None,
            training_mode: TrainingMode::Hard,
        }
    }
}

#[derive(Module, Debug)]
pub struct StochasticAttentionStaticWindow {
    q_idx: Tensor<1, Int>,
    k_idx: Tensor<1, Int>,
    v_idx: Tensor<1, Int>,
    inv_scale: f32,
    band_bias: Param<Tensor<5>>, // [H, N, 2w+1]
    temperature: f32,
    half_width: usize,
    nhead: usize,
    dk: usize,
    window_indices: Tensor<1, Int>, // [N * bw]
    score_mode: StochasticMul,
    seq_length: usize,
}

impl StochasticAttentionStaticWindow {
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

    fn calc_qkv_hard(&self, x: Tensor<4>) -> (Tensor<5>, Tensor<5>, Tensor<5>) {
        let q = x.clone().select(3, self.q_idx.clone()).unsqueeze_dim(3);
        let k = self.local_window(x.clone().select(3, self.k_idx.clone()));
        //let v = self.local_window(x.select(3, self.v_idx.clone()));

        //(q, k, v)
        (q, k, self.local_window(x))
    }

    fn calc_scores(&self, q: Tensor<5>, k_win: Tensor<5>) -> Tensor<5> {
        let scores = q.matmul(k_win) * self.inv_scale + self.band_bias.val();
        match self.score_mode {
            StochasticMul::Softmax => softmax(scores, 4),
            StochasticMul::Sinkhorn => {
                sinkhorn(scores, self.temperature, NormalizationMode::Double)
            }
            StochasticMul::None => scores,
        } // [B,N,H,1,bw]
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.nhead;

        let x = x.reshape([b, n, h, dk]);

        let (q, k, v) = self.calc_qkv_hard(x);
        let p = self.calc_scores(q, k);
        let out = v.matmul(p.transpose());

        out.reshape([b, n, e])
    }
}

impl StochasticAttentionStaticWindowConfig {
    pub fn init(&self, device: &Device) -> StochasticAttentionStaticWindow {
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
            Tensor::<4>::random([1, 1, dk, dk], Distribution::Normal(0.0, logit_std), device)
                .argmax(2)
                .reshape([dk])
        };

        StochasticAttentionStaticWindow {
            band_bias: Param::from_tensor(Tensor::<5>::zeros(
                [1, self.seq_length, self.nhead, 1, window],
                device,
            ))
            .set_require_grad(true),
            temperature: self.temperature,
            half_width: w,
            nhead: self.nhead,
            q_idx: init_logits(),
            k_idx: init_logits(),
            v_idx: init_logits(),
            dk,
            inv_scale: 1.0 / (dk as f32).sqrt(),
            window_indices: window_indices.clone().reshape([n * window]),
            seq_length: self.seq_length,
            score_mode: self.score_mode,
        }
    }
}
