use burn::{config::Config, module::Module, prelude::*, tensor::activation::softmax};

use crate::conv::{
    DepthwiseConvBnAct, DepthwiseConvBnActConfig, PointwiseConvBn, PointwiseConvBnConfig,
};

#[derive(Module, Debug)]
pub struct CGAHead<B: Backend> {
    q_proj: PointwiseConvBn<B>,
    k_proj: PointwiseConvBn<B>,
    v_proj: PointwiseConvBn<B>,
    q_token_interaction: DepthwiseConvBnAct<B>,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
}

#[derive(Config, Debug)]
pub struct CGAHeadConfig {
    pub in_channels: usize,
    pub q_dim: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    #[config(default = 3)]
    pub token_kernel_size: usize,
}

impl CGAHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CGAHead<B> {
        CGAHead {
            q_proj: PointwiseConvBnConfig::new(self.in_channels, self.q_dim).init(device),
            k_proj: PointwiseConvBnConfig::new(self.in_channels, self.k_dim).init(device),
            v_proj: PointwiseConvBnConfig::new(self.in_channels, self.v_dim).init(device),
            q_token_interaction: DepthwiseConvBnActConfig::new(self.q_dim)
                .with_kernel_size(self.token_kernel_size)
                .init(device),
            q_dim: self.q_dim,
            k_dim: self.k_dim,
            v_dim: self.v_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct CascadedGroupAttention<B: Backend> {
    heads: Vec<CGAHead<B>>,
    proj: PointwiseConvBn<B>,
    num_heads: usize,
}

#[derive(Config, Debug)]
pub struct CascadedGroupAttentionConfig {
    pub dim: usize,
    pub num_heads: usize,
    #[config(default = 3)]
    pub token_kernel_size: usize,
    #[config(default = 0)]
    pub q_dim: usize,
    #[config(default = 0)]
    pub k_dim: usize,
    #[config(default = 0)]
    pub v_dim: usize,
}

impl CascadedGroupAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CascadedGroupAttention<B> {
        assert!(self.num_heads > 0);
        assert!(self.dim.is_multiple_of(self.num_heads));

        let split_dim = self.dim / self.num_heads;

        let q_dim = if self.q_dim == 0 {
            split_dim
        } else {
            self.q_dim
        };
        let k_dim = if self.k_dim == 0 {
            split_dim
        } else {
            self.k_dim
        };
        let v_dim = if self.v_dim == 0 {
            split_dim
        } else {
            self.v_dim
        };

        let mut heads = Vec::with_capacity(self.num_heads);
        for _ in 0..self.num_heads {
            heads.push(
                CGAHeadConfig::new(split_dim, q_dim, k_dim, v_dim)
                    .with_token_kernel_size(self.token_kernel_size)
                    .init(device),
            );
        }

        CascadedGroupAttention {
            heads,
            proj: PointwiseConvBnConfig::new(self.num_heads * v_dim, self.dim).init(device),
            num_heads: self.num_heads,
        }
    }
}

impl<B: Backend> CascadedGroupAttention<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();

        let splits = x.chunk(self.num_heads, 1);
        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(self.num_heads);
        let mut prev_out: Option<Tensor<B, 4>> = None;

        for (i, split) in splits.into_iter().enumerate() {
            let head = &self.heads[i];

            let head_input = match prev_out.as_ref() {
                Some(prev) => split + prev.clone(),
                None => split,
            };

            let q = head
                .q_token_interaction
                .forward(head.q_proj.forward(head_input.clone()));
            let k = head.k_proj.forward(head_input.clone());
            let v = head.v_proj.forward(head_input);

            let [b, q_c, h, w] = q.dims();
            let n = h * w;

            let q = q.reshape([b, q_c, n]).swap_dims(1, 2); // [B, N, Cq]
            let k = k.reshape([b, head.k_dim, n]); // [B, Ck, N]
            let v = v.reshape([b, head.v_dim, n]).swap_dims(1, 2); // [B, N, Cv]

            let scale = 1.0f64 / (head.k_dim as f64).sqrt();
            let attn = softmax(q.matmul(k) * scale, 2); // softmax over last dim N

            let out = attn
                .matmul(v) // [B, N, Cv]
                .swap_dims(1, 2) // [B, Cv, N]
                .reshape([b, head.v_dim, h, w]);

            prev_out = Some(out.clone());
            outputs.push(out);
        }

        let x = Tensor::cat(outputs, 1);
        let x = self.proj.forward(x);

        x + residual
    }
}
