use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::Tensor,
    tensor::{Distribution, TensorData},
};

use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct LearnedPermuter<B: Backend> {
    scores_u: Param<Tensor<B, 3>>, // [N, rank]
    scores_v: Param<Tensor<B, 3>>, // [N, rank]
    bias_proj: Linear<B>,
    temperature: f32,
}

#[derive(Config, Debug)]
pub struct LearnedPermuterConfig {
    pub embed_dim: usize,
    pub seq_length: usize,
    pub layer_num: usize,
    pub temperature: f32,

    #[config(default = 8)]
    pub rank: usize,
}

impl LearnedPermuterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LearnedPermuter<B> {
        let (u, v) = Self::layer_scores(self.seq_length, self.rank, self.layer_num, device);
        LearnedPermuter {
            scores_u: Param::from_tensor(u.unsqueeze_dim(0)).set_require_grad(true),
            scores_v: Param::from_tensor(v.transpose().unsqueeze_dim(0)).set_require_grad(true),
            bias_proj: LinearConfig::new(self.embed_dim, self.seq_length).init(device),
            temperature: self.temperature,
        }
    }

    /// Factorize the cyclic-shift identity into U, V such that U @ V^T ≈ shift matrix.
    /// Uses the first `rank` DCT basis vectors as a structured low-rank initialisation
    /// so each layer still starts from a different prior.
    fn layer_scores<B: Backend>(
        n: usize,
        rank: usize,
        layer_num: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        use std::f32::consts::PI;
        let shift = layer_num % n;

        let mut u_data = vec![0.0f32; n * rank];
        let mut v_data = vec![0.0f32; n * rank];

        for k in 0..rank {
            let scale = if k == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
            for i in 0..n {
                let basis = scale * (PI * k as f32 * (2 * i + 1) as f32 / (2.0 * n as f32)).cos();
                u_data[i * rank + k] = basis;
                let j = (i + shift) % n;
                v_data[j * rank + k] = basis;
            }
        }

        let noise = |data: Vec<f32>| {
            let base = Tensor::<B, 2>::from_data(TensorData::new(data, [n, rank]), device);
            let noise = Tensor::random([n, rank], Distribution::Normal(0.0, 0.02), device);
            base + noise
        };

        (noise(u_data), noise(v_data))
    }
}

impl<B: Backend> LearnedPermuter<B> {
    fn sinkhorn(&self, s: Tensor<B, 3>) -> Tensor<B, 3> {
        let s = s / self.temperature;
        let s_centered = s.clone() - s.max_dim(2);
        let s_exp = s_centered.exp();
        s_exp.clone() / s_exp.sum_dim(2)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, _] = x.dims();

        let summary = x.clone().mean_dim(1).squeeze_dim::<2>(1);
        let bias = self.bias_proj.forward(summary).reshape([b, n, 1]);

        // Reconstruct [1, N, N] from low-rank factors then add content bias
        let scores = self.scores_u.val()
            .matmul(self.scores_v.val())  // [1, N, N]
            + bias; // [B, N, N]

        let p = self.sinkhorn(scores);
        p.matmul(x)
    }
}
