use burn::{
    module::{Module, Param},
    prelude::{Device, Tensor},
    tensor::{Distribution, activation::softmax},
};

use crate::attention::{
    NormalizationMode, StochasticMul, StochasticSelect, TrainingMode, calc_qkv_hard, calc_qkv_soft,
    sinkhorn,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StochasticAttentionConfig {
    pub embed_dim: usize,
    pub nhead: usize,
    pub temperature: f32,
    #[serde(default)]
    pub stoch_mode: StochasticSelect,
    #[serde(default)]
    pub norm_mode: NormalizationMode,
    #[serde(default)]
    pub score_mode: StochasticMul,
    #[serde(default)]
    pub training_mode: TrainingMode,
}

impl StochasticAttentionConfig {
    pub fn new(embed_dim: usize, nhead: usize, temperature: f32) -> Self {
        Self {
            embed_dim,
            nhead,
            temperature,
            stoch_mode: StochasticSelect::Qkv,
            norm_mode: NormalizationMode::Single,
            score_mode: StochasticMul::Sinkhorn,
            training_mode: TrainingMode::Hard,
        }
    }
}

#[derive(Module, Debug)]
pub struct StochasticAttention {
    q_mat: Param<Tensor<4>>,
    k_mat: Param<Tensor<4>>,
    v_mat: Param<Tensor<4>>,
    inv_scale: f32,
    temperature: f32,
    nhead: usize,
    dk: usize,
    stoch_mode: StochasticSelect,
    norm_mode: NormalizationMode,
    score_mode: StochasticMul,
    train_mode: TrainingMode,
    
}

impl StochasticAttention {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self.train_mode {
            TrainingMode::Soft => self.forward_soft(x),
            TrainingMode::Hard | TrainingMode::Mixed => self.forward_hard(x),
        }
    }

    fn calc_scores(&self, q: Tensor<4>, k: Tensor<4>) -> Tensor<4> {
        // [B,H,N,dk] x [B,H,dk,N] -> [B,H,N,N], all-pairs scores
        let scores = q.matmul(k.transpose()) * self.inv_scale;
        match self.score_mode {
            StochasticMul::Softmax => softmax(scores, 3),
            StochasticMul::Sinkhorn => sinkhorn(scores, self.temperature, self.norm_mode),
            StochasticMul::None => scores,
        }
    }

    pub fn forward_hard(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.nhead;

        let x = x.reshape([b, n, h, dk]);
        let (q, k, v) = calc_qkv_hard(
            x,
            self.stoch_mode,
            self.q_mat.val(),
            self.k_mat.val(),
            self.v_mat.val(),
        );

        // [B,N,H,dk] -> [B,H,N,dk] so heads batch like independent attentions
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let p = self.calc_scores(q, k);
        let out = p.matmul(v); // [B,H,N,dk]

        out.swap_dims(1, 2).reshape([b, n, e])
    }

    pub fn forward_soft(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.nhead;

        let x = x.reshape([b, n, h, dk]);
        let (w_q, w_k, w_v) = calc_qkv_soft(
            self.temperature,
            self.stoch_mode,
            self.norm_mode,
            self.q_mat.val(),
            self.k_mat.val(),
            self.v_mat.val(),
        );

        let q = x.clone().matmul(w_q).swap_dims(1, 2);
        let k = x.clone().matmul(w_k).swap_dims(1, 2);
        let v = x.matmul(w_v).swap_dims(1, 2);

        let p = self.calc_scores(q, k);
        let out = p.matmul(v); // [B,H,N,dk]

        out.swap_dims(1, 2).reshape([b, n, e])
    }
}

impl StochasticAttentionConfig {
    pub fn init(&self, device: &Device) -> StochasticAttention {
        let dk = self.embed_dim / self.nhead;
        let logit_std = (1.0 / dk as f64).sqrt();

        let init_logits = || {
            Param::from_tensor(Tensor::<4>::random(
                [1, 1, dk, dk],
                Distribution::Normal(0.0, logit_std),
                device,
            ))
            .set_require_grad(true)
        };

        StochasticAttention {
            q_mat: init_logits(),
            k_mat: init_logits(),
            v_mat: init_logits(),
            dk,
            inv_scale: 1.0 / (dk as f32).sqrt(),
            temperature: self.temperature,
            nhead: self.nhead,
            stoch_mode: self.stoch_mode,
            norm_mode: self.norm_mode,
            score_mode: self.score_mode,
            train_mode: self.training_mode,
        }
    }
}
