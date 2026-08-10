use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig, Softplus, SoftplusConfig},
    prelude::{Device, Tensor},
    tensor::linalg::l2_norm,
};

#[derive(Config, Debug)]
pub struct DSTHAConfig {
    pub d_model: usize,
    pub n_heads: usize,
    #[config(default = 2)]
    pub sinkhorn_iters: usize,
    #[config(default = 1e-6)]
    pub epsilon: f32,
    #[config(default = 0.0)]
    pub init_alpha: f32,
    #[config(default = 1.0)]
    pub init_gamma: f32,
}

#[derive(Module, Debug)]
pub struct DSTHA {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    delta_proj: Linear,
    softplus: Softplus,
    raw_gamma: Param<Tensor<1>>,
    alpha: Param<Tensor<1>>,
    n_heads: usize,
    d_head: usize,
    sinkhorn_iters: usize,
    epsilon: f32,
}

impl DSTHA {
    fn m(&self) -> Tensor<1> {
        (1.0_f32 - self.alpha.val()).sqrt()
    }

    fn gamma(&self) -> Tensor<1> {
        //self.softplus.forward(self.raw_gamma.val()) + 1e-4
        self.raw_gamma.val()
    }

    fn phi(&self, x: Tensor<4>, scale_b: Tensor<4>, gamma_b: Tensor<4>, m: Tensor<1>) -> Tensor<4> {
        let [batch, heads, seq, d] = x.dims();
        let device = x.device();

        let m_b = m.reshape([1, heads, 1, 1]);

        let shifted = x.clone() / (d as f32).powf(0.25) + m_b; // x_i/d^(1/4) + m
        let psi_raw = shifted.clone() * shifted; // squared, per-coordinate

        let psi_sq_sum = (psi_raw.clone() * psi_raw.clone()).sum_dim(3);
        let psi_norm = psi_sq_sum.sqrt() + 1e-6;

        // delta(x): learnable per-token scale, applied to the *raw* token
        // vector x (same x that psi_m is built from), softplus'd positive.
        let delta = self.softplus.forward(self.delta_proj.forward(x)) + 1e-4; // [b,h,seq,1]

        let psi = delta * psi_raw / psi_norm; // psi_hat(x), broadcasts over d

        let gamma_term = Tensor::<4>::ones([batch, heads, seq, 1], &device) * gamma_b;
        let feat = Tensor::cat(vec![psi, gamma_term], 3); // [psi_hat(x); gamma]

        feat * scale_b // scale by sqrt(e^alpha/2)
    }

    /// phi_q, phi_k: [batch, heads, seq, d_head+1]
    /// returns (u, w): each [batch, heads, seq, 1]
    fn sinkhorn(&self, phi_q: Tensor<4>, phi_k: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        //let [batch, heads, seq, _] = phi_q.dims();
        //let device = phi_q.device();

        //let mut u = Tensor::<4>::ones([batch, heads, seq, 1], &device);
        //let mut w = Tensor::<4>::ones([batch, heads, seq, 1], &device);

        let phi_q_t = phi_q.clone().transpose(); // [batch, heads, d+1, seq]
        let phi_k_t = phi_k.clone().transpose();

        let mut u = 1.0_f32 / (l2_norm(phi_q.clone(), 3));
        let mut w = 1.0_f32 / (l2_norm(phi_k.clone(), 3));

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

    fn project_and_featurize(&self, x: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [batch, seq, _] = x.dims();
        let (h, d) = (self.n_heads, self.d_head);

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([batch, seq, h, d]).swap_dims(1, 2);
        let k = k.reshape([batch, seq, h, d]).swap_dims(1, 2);
        let v = v.reshape([batch, seq, h, d]).swap_dims(1, 2);

        let m = self.m(); // [heads]
        let alpha = 1.0_f32 - m.clone().square(); // [heads]
        let scale = (alpha.exp() / 2.0).sqrt(); // sqrt(e^alpha / 2), [heads]

        let scale_b = scale.reshape([1, h, 1, 1]);
        let gamma_b = self.gamma().reshape([1, h, 1, 1]);

        (
            self.phi(q, scale_b.clone(), gamma_b.clone(), m.clone()),
            self.phi(k, scale_b, gamma_b, m),
            v,
        )
    }

    fn apply_output(&self, phi_q: Tensor<4>, phi_k: Tensor<4>, v: Tensor<4>) -> Tensor<3> {
        let [batch, h, seq, _] = v.dims();
        let d = self.d_head;

        let context = phi_k.swap_dims(2, 3).matmul(v); // [b,h,d+1,d_head]
        let out = phi_q.matmul(context); // [b,h,seq,d_head]

        let out = out.swap_dims(1, 2).reshape([batch, seq, h * d]);
        self.out_proj.forward(out)
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let (phi_q, phi_k, v) = self.project_and_featurize(x);
        let (u, w) = self.sinkhorn(phi_q.clone(), phi_k.clone());

        let phi_q_scaled = phi_q * u;
        let phi_k_scaled = phi_k * w;

        self.apply_output(phi_q_scaled, phi_k_scaled, v)
    }

    /// L_DS = ||r-1||^2 + ||c-1||^2.
    pub fn marginals(&self, x: Tensor<3>) -> (Tensor<4>, Tensor<4>) {
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
    pub fn init(&self, device: &Device) -> DSTHA {
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

        let alpha = Tensor::<1>::full([self.n_heads], self.init_alpha, device);

        let raw_gamma = Tensor::<1>::full([self.n_heads], self.init_gamma, device);

        let init_logits = || LinearConfig::new(self.d_model, self.d_model).init(device);
        DSTHA {
            q_proj: init_logits(),
            k_proj: init_logits(),
            v_proj: init_logits(),
            out_proj: init_logits(),
            softplus: SoftplusConfig::new().with_beta(0.1).init(),
            delta_proj: LinearConfig::new(d_head, 1).init(device),
            raw_gamma: Param::from_tensor(raw_gamma),
            alpha: Param::from_tensor(alpha),
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
    use burn::tensor::{Device, Distribution};

    fn device() -> Device {
        Device::flex()
    }

    #[test]
    fn forward_runs_and_marginals_converge_to_one() {
        let device = device();
        let config = DSTHAConfig::new(16, 4).with_sinkhorn_iters(10);
        let model = config.init(&device);

        let x = Tensor::<3>::random([2, 20, 16], Distribution::Normal(0.0, 1.0), &device);

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
