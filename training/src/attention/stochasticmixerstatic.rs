use burn::{
    module::Module,
    prelude::Tensor,
    tensor::{Distribution, Int, activation::softmax, backend::Backend},
};

use crate::attention::{NormalizationMode, StochasticMul, sinkhorn};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StochasticAttentionStaticConfig {
    pub embed_dim: usize,
    pub nhead: usize,
    pub temperature: f32,
    #[serde(default)]
    pub score_mode: StochasticMul,
}

impl StochasticAttentionStaticConfig {
    pub fn new(embed_dim: usize, nhead: usize, temperature: f32) -> Self {
        Self {
            embed_dim,
            nhead,
            temperature,
            score_mode: StochasticMul::Softmax,
        }
    }
}

#[derive(Module, Debug)]
pub struct StochasticAttentionStatic<B: Backend> {
    q_idx: Tensor<B, 1, Int>,
    k_idx: Tensor<B, 1, Int>,
    v_idx: Tensor<B, 1, Int>,
    inv_scale: f32,
    temperature: f32,
    nhead: usize,
    dk: usize,
    score_mode: StochasticMul,
}

impl<B: Backend> StochasticAttentionStatic<B> {
    fn calc_scores(&self, q: Tensor<B, 4>, k: Tensor<B, 4>) -> Tensor<B, 4> {
        // [B,H,N,dk] x [B,H,dk,N] -> [B,H,N,N], all-pairs scores
        let scores = q.matmul(k.transpose()) * self.inv_scale;
        match self.score_mode {
            StochasticMul::Softmax => softmax(scores, 3),
            StochasticMul::Sinkhorn => {
                sinkhorn(scores, self.temperature, NormalizationMode::Double)
            }
            StochasticMul::None => scores,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.nhead;

        let x = x.reshape([b, n, h, dk]).swap_dims(1, 2);
        let q = x.clone().select(3, self.q_idx.clone());
        let k = x.clone().select(3, self.k_idx.clone());
        //let v = x.select(3, self.v_idx.clone());

        // [B,N,H,dk] -> [B,H,N,dk] so heads batch like independent attentions
        //let q = q.swap_dims(1, 2);
        //let k = k.swap_dims(1, 2);
        //let v = v.swap_dims(1, 2);

        let p = self.calc_scores(q, k);
        //let out = p.matmul(v); // [B,H,N,dk]
        let out = p.matmul(x); // [B,H,N,dk]

        out.swap_dims(1, 2).reshape([b, n, e])
    }
}

impl StochasticAttentionStaticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StochasticAttentionStatic<B> {
        let dk = self.embed_dim / self.nhead;
        let logit_std = (1.0 / dk as f64).sqrt();

        //x.clone().select(3, mat.argmax(2).reshape([dk]))
        let init_logits = || {
            Tensor::<B, 4>::random([1, 1, dk, dk], Distribution::Normal(0.0, logit_std), device)
                .argmax(2)
                .reshape([dk])
        };

        StochasticAttentionStatic {
            q_idx: init_logits(),
            k_idx: init_logits(),
            v_idx: init_logits(),
            dk,
            inv_scale: 1.0 / (dk as f32).sqrt(),
            temperature: self.temperature,
            nhead: self.nhead,
            score_mode: self.score_mode,
        }
    }
}
