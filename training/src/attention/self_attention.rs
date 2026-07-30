use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Tensor,
    tensor::{activation::softmax, backend::Backend},
};

#[derive(Config, Debug)]
pub struct SelfAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
}

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    n_heads: usize,
    d_head: usize,
    scale: f32,
}

impl<B: Backend> SelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (scores, v) = self.scores(x);

        let scores = softmax(scores, 3);

        self.apply_output(scores, v)
    }

    pub fn scores(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, _] = x.dims();
        let (h, d) = (self.n_heads, self.d_head);

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([batch, seq, h, d]).swap_dims(1, 2);
        let k = k.reshape([batch, seq, h, d]).swap_dims(1, 2);
        let v = v.reshape([batch, seq, h, d]).swap_dims(1, 2);

        (q.matmul(k.transpose()) / self.scale, v)
    }

    pub fn apply_output(&self, scores: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _, seq, _] = scores.dims();
        let (h, d) = (self.n_heads, self.d_head);

        let out = scores.matmul(v);
        let out = out.swap_dims(1, 2).reshape([batch, seq, h * d]);
        self.out_proj.forward(out)
    }
}

impl SelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SelfAttention<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads,
        );

        let d_head = self.d_model / self.n_heads;
        let init_logits = || LinearConfig::new(self.d_model, self.d_model).init(device);

        SelfAttention {
            n_heads: self.n_heads,
            d_head,
            q_proj: init_logits(),
            k_proj: init_logits(),
            v_proj: init_logits(),
            out_proj: init_logits(),
            scale: (d_head as f32).sqrt(),
        }
    }
}
