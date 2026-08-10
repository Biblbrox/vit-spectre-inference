
use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    tensor::{Device, Distribution},
};

use crate::{kernels::monarch_fused_reference, linear::optimal_block_count, utils::gcd};

#[derive(Module, Debug)]
pub struct MonarchLinear {
    left: Param<Tensor<3>>,         // [a, c, b]
    right: Param<Tensor<3>>,        // [c, d, a]
    bias: Option<Param<Tensor<3>>>, // [1, 1, out_features]
    in_features: usize,
    out_features: usize,
    a: usize, // in block count
    b: usize, // in block size  (a*b = in_features)
    d: usize, // out block size (c*d = out_features)
    
}

#[derive(Config, Debug)]
pub struct MonarchLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    #[config(default = true)]
    pub bias: bool,
}

impl MonarchLinearConfig {
    pub fn init(&self, device: &Device) -> MonarchLinear {
        let a = optimal_block_count(self.in_features, self.out_features);
        let b = self.in_features / a;
        let d = self.out_features / a;

        assert!(
            a > 1,
            "No valid factorization found for in={} out={} — \
             gcd={} has no useful divisors. Consider adjusting dimensions.",
            self.in_features,
            self.out_features,
            gcd(self.in_features, self.out_features)
        );

        let std_r = (2.0 / self.in_features as f64).sqrt();
        let left =
            Tensor::<3>::random([a, a, b], Distribution::Normal(0.0, 1.0 / a as f64), device);
        let right = Tensor::<3>::random([a, d, a], Distribution::Normal(0.0, std_r), device);

        MonarchLinear {
            left: Param::from_tensor(left).set_require_grad(true),
            right: Param::from_tensor(right).set_require_grad(true),
            bias: if self.bias {
                Some(
                    Param::from_tensor(Tensor::<3>::zeros([1, 1, self.out_features], device))
                        .set_require_grad(true),
                )
            } else {
                None
            },
            in_features: self.in_features,
            out_features: self.out_features,
            a,
            b,
            d,
        }
    }
}

impl MonarchLinear {
    /// x: [B, N, in_features]
    /// output: [B, N, out_features]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        monarch_fused_reference(
            x,
            self.left.val(),
            self.right.val(),
            self.a,
            self.b,
            self.bias.as_ref().map(|b| b.val()), //match self.bias.clone() {
                                                 //    Some(bias) => Some(bias.val()),
                                                 //    None => None,
                                                 //},
        )
    }
}
