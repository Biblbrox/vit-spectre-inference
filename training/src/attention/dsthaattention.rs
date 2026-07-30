use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::Tensor,
    tensor::backend::Backend,
};

#[derive(Config, Debug)]
pub struct DSTHAConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = 3)]
    pub sinkhorn_iters: usize,
    #[config(default = 1e-6)]
    pub epsilon: f32,
    #[config(default = 0.0)]
    pub init_alpha: f32,
    #[config(default = 1.0)]
    pub init_gamma: f32,
}

#[derive(Module, Debug)]
pub struct DSTHA<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    delta_proj: Linear<B>,
    raw_gamma: Param<Tensor<B, 1>>,
    raw_m: Param<Tensor<B, 1>>,
    n_heads: usize,
    d_head: usize,
    sinkhorn_iters: usize,
    epsilon: f32,
}

fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let abs_x = x.clone().abs();
    let max_x0 = (x + abs_x.clone()) / 2.0; // max(x,0), via (x+|x|)/2
    let log_term = (abs_x * -1.0).exp() + 1.0; // 1 + exp(-|x|), always in (1,2]
    max_x0 + log_term.log()
}

fn inv_softplus_init(target: f32, eps: f32) -> f32 {
    ((target - eps).exp() - 1.0).ln()
}

impl<B: Backend> DSTHA<B> {
    fn m(&self) -> Tensor<B, 1> {
        softplus(self.raw_m.val()) + 1e-4
    }

    fn gamma(&self) -> Tensor<B, 1> {
        softplus(self.raw_gamma.val()) + 1e-4
    }

    fn phi(
        &self,
        x: Tensor<B, 4>,
        scale_b: Tensor<B, 4>,
        gamma_b: Tensor<B, 4>,
        m: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let [batch, heads, seq, d] = x.dims();
        let device = x.device();

        let m_b = m.reshape([1, heads, 1, 1]);

        let shifted = x.clone() / (d as f32).powf(0.25) + m_b; // x_i/d^(1/4) + m
        let psi_raw = shifted.clone() * shifted; // squared, per-coordinate

        let psi_sq_sum = (psi_raw.clone() * psi_raw.clone()).sum_dim(3);
        let psi_norm = psi_sq_sum.sqrt() + 1e-6;

        // delta(x): learnable per-token scale, applied to the *raw* token
        // vector x (same x that psi_m is built from), softplus'd positive.
        let delta = softplus(self.delta_proj.forward(x)) + 1e-4; // [b,h,seq,1]

        let psi = delta * psi_raw / psi_norm; // psi_hat(x), broadcasts over d

        let gamma_term = Tensor::<B, 4>::ones([batch, heads, seq, 1], &device) * gamma_b;
        let feat = Tensor::cat(vec![psi, gamma_term], 3); // [psi_hat(x); gamma]

        feat * scale_b // scale by sqrt(e^alpha/2)
    }

    /// phi_q, phi_k: [batch, heads, seq, d_head+1]
    /// returns (u, w): each [batch, heads, seq, 1]
    fn sinkhorn(&self, phi_q: Tensor<B, 4>, phi_k: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, heads, seq, _] = phi_q.dims();
        let device = phi_q.device();

        let mut u = Tensor::<B, 4>::ones([batch, heads, seq, 1], &device);
        let mut w = Tensor::<B, 4>::ones([batch, heads, seq, 1], &device);

        let phi_q_t = phi_q.clone().transpose(); // [batch, heads, d+1, seq]
        let phi_k_t = phi_k.clone().transpose();

        for _ in 0..self.sinkhorn_iters {
            let qt_u = phi_q_t.clone().matmul(u.clone()); // [b,h,d+1,1]
            let k_qtu = phi_k.clone().matmul(qt_u); // [b,h,seq,1]
            w = (k_qtu + self.epsilon).recip();

            let kt_w = phi_k_t.clone().matmul(w.clone()); // [b,h,d+1,1]
            let q_ktw = phi_q.clone().matmul(kt_w); // [b,h,seq,1]
            u = (q_ktw + self.epsilon).recip();

            let mean_u = u.clone().mean_dim(2);
            let mean_w = w.clone().mean_dim(2);
            let g = (mean_w / mean_u).sqrt();
            u = u * g.clone();
            w = w / g;
        }

        (u, w)
    }

    fn project_and_featurize(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq, _] = x.dims();
        let (h, d) = (self.n_heads, self.d_head);

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([batch, seq, h, d]).swap_dims(1, 2);
        let k = k.reshape([batch, seq, h, d]).swap_dims(1, 2);
        let v = v.reshape([batch, seq, h, d]).swap_dims(1, 2);

        let m = self.m(); // [heads]
        let m_sq = m.clone() * m.clone();
        let alpha = m_sq * -1.0 + 1.0; // alpha = 1 - m^2, [heads]
        let scale = (alpha.exp() / 2.0).sqrt(); // sqrt(e^alpha / 2), [heads]

        let scale_b = scale.reshape([1, h, 1, 1]);
        let gamma_b = self.gamma().reshape([1, h, 1, 1]);

        (
            self.phi(q, scale_b.clone(), gamma_b.clone(), m.clone()),
            self.phi(k, scale_b, gamma_b, m),
            v,
        )
    }

    fn apply_output(
        &self,
        phi_q: Tensor<B, 4>,
        phi_k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, h, seq, _] = v.dims();
        let d = self.d_head;

        let context = phi_k.swap_dims(2, 3).matmul(v); // [b,h,d+1,d_head]
        let out = phi_q.matmul(context); // [b,h,seq,d_head]

        let out = out.swap_dims(1, 2).reshape([batch, seq, h * d]);
        self.out_proj.forward(out)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (phi_q, phi_k, v) = self.project_and_featurize(x);
        let (u, w) = self.sinkhorn(phi_q.clone(), phi_k.clone());

        let phi_q_scaled = phi_q * u;
        let phi_k_scaled = phi_k * w;

        self.apply_output(phi_q_scaled, phi_k_scaled, v)
    }

    /// L_DS = ||r-1||^2 + ||c-1||^2.
    pub fn marginals(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (phi_q, phi_k, _v) = self.project_and_featurize(x);
        let (u, w) = self.sinkhorn(phi_q.clone(), phi_k.clone());

        let phi_q_t = phi_q.clone().swap_dims(2, 3);
        let phi_k_t = phi_k.clone().swap_dims(2, 3);

        let kt_w = phi_k_t.matmul(w.clone());
        let r = u.clone() * phi_q.matmul(kt_w); // row sums

        let qt_u = phi_q_t.matmul(u);
        let c = w * phi_k.matmul(qt_u); // col sums

        (r, c)
    }
}

impl DSTHAConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DSTHA<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads,
        );
        assert!(
            self.init_alpha <= 1.0,
            "init_alpha must be <= 1 so that m = sqrt(1 - alpha) is real"
        );
        let d_head = self.d_model / self.n_heads;

        let m0 = (1.0 - self.init_alpha).sqrt().max(1e-3);
        let eps_m = 1e-4_f32;
        let raw0 = ((m0 - eps_m).exp() - 1.0).ln();

        let raw_m = Tensor::<B, 1>::full([self.n_heads], raw0, device);

        let raw_gamma0 = inv_softplus_init(self.init_gamma.max(1e-3), eps_m);
        let raw_gamma = Tensor::<B, 1>::full([self.n_heads], raw_gamma0, device);

        let init_logits = || LinearConfig::new(self.d_model, self.d_model).init(device);
        DSTHA {
            q_proj: init_logits(),
            k_proj: init_logits(),
            v_proj: init_logits(),
            out_proj: init_logits(),
            delta_proj: LinearConfig::new(d_head, 1).init(device),
            raw_gamma: Param::from_tensor(raw_gamma),
            raw_m: Param::from_tensor(raw_m),
            n_heads: self.n_heads,
            d_head,
            sinkhorn_iters: self.sinkhorn_iters,
            epsilon: self.epsilon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::tensor::Distribution;

    type B = Flex<f32>;

    #[test]
    fn forward_runs_and_marginals_converge_to_one() {
        let device = Default::default();
        let config = DSTHAConfig::new(16, 4).with_sinkhorn_iters(10);
        let model = config.init::<B>(&device);

        let x = Tensor::<B, 3>::random([2, 20, 16], Distribution::Normal(0.0, 1.0), &device);

        let out = model.forward(x.clone());
        assert_eq!(out.dims(), [2, 20, 16]);

        let (r, c) = model.marginals(x);
        let r_data: Vec<f32> = r.into_data().to_vec().unwrap();
        let c_data: Vec<f32> = c.into_data().to_vec().unwrap();

        for v in r_data.iter().chain(c_data.iter()) {
            assert!((v - 1.0).abs() < 0.05, "marginal {} not close to 1", v);
        }
    }
}
