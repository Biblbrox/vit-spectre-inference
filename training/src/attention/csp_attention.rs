use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

/// CSP attention implementation according to https://arxiv.org/pdf/2410.10914
#[derive(Config, Debug)]
pub struct CspConfig {
    pub d_model: usize,
    pub seq_length: usize,
    pub group_size: usize,
    #[config(default = 1.0)]
    pub temperature: f32,
}

#[derive(Module, Debug)]
pub struct Csp<B: Backend> {
    v_proj: Linear<B>,
    out_proj: Linear<B>,

    shift_idx: Tensor<B, 3, Int>,

    d_model: usize,
    seq_length: usize,
    group_size: usize,
    temperature: f32,
}

fn build_shift_idx<B: Backend>(seq_length: usize, d_model: usize) -> Tensor<B, 3, Int> {
    let mut idx = Vec::with_capacity(seq_length * d_model);

    for t in 0..seq_length {
        for c in 0..d_model {
            let shift = ((c + 1) * (c + 1) - 1) % seq_length;
            idx.push(((t + shift) % seq_length) as i64);
        }
    }

    let data = TensorData::new(idx, &[1, d_model, seq_length]);
    Tensor::<B, 3, Int>::from(data)
}

impl CspConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Csp<B> {
        assert!(self.d_model > 0, "d_model must be > 0");
        assert!(self.seq_length > 0, "seq_len must be > 0");
        assert!(self.group_size > 0, "group_size must be > 0");

        let shift_idx = build_shift_idx(self.seq_length, self.d_model);
        let d = self.d_model;

        Csp {
            v_proj: LinearConfig::new(d, d).with_bias(false).init(device),
            out_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            d_model: self.d_model,
            seq_length: self.seq_length,
            group_size: self.group_size,
            temperature: self.temperature,
            shift_idx,
        }
    }
}

impl<B: Backend> Csp<B> {
    fn pad_tokens(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        let remainder = n % self.group_size;
        if remainder == 0 {
            return x;
        }

        let pad = self.group_size - remainder;
        let zeros = Tensor::<B, 3>::zeros([b, pad, d], &x.device());
        let padded = Tensor::cat(vec![x, zeros], 1);
        padded
    }

    /// Shift each channel independently along the token axis.
    /// We do it with gather because shifts are hardcoded at init
    /// and don't change after that.
    /// Input:  [B, N, D]
    /// Output: [B, N, D]
    fn channelwise_shift(&self, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let b = v.dims()[0];

        let idx = self
            .shift_idx
            .clone()
            .unsqueeze::<3>() // [1, N, D]
            .repeat_dim(0, b); // [B, N, D]
        v.gather(1, idx)
    }

    /// Apply a gsort operator from the paper.
    /// Input: [B, N, D]
    /// Output: [B, N, D]
    fn gsort_operator(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        let g = n / self.group_size;
        // [B, d, group_size, num_groups]
        let groups = x.reshape([b, g, self.group_size, d]).swap_dims(1, 3);
        let sort_idx = groups.clone().argsort(2);
        groups
            .gather(2, sort_idx)
            .swap_dims(1, 3)
            .reshape([b as i64, -1, d as i64])
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        debug_assert_eq!(d, self.d_model, "input dim must match d_model");

        let v = self.v_proj.forward(x); // [B, N, D]
        let v = self.channelwise_shift(v); // [B, N, D]
        let v_padded = self.pad_tokens(v);

        let out = self.gsort_operator(v_padded);

        // We do slice anyway even if the pad is zero to reduce cache misses
        self.out_proj.forward(out.slice([0..b, 0..(n), 0..d]))
    }
}
