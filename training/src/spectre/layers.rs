
use burn::{
    config::Config,
    module::Module,
    tensor::{Device, Shape, Tensor, TensorData},
};

use crate::spectre::transform::build_dct_projection;

/// A parameter-free linear layer whose weight matrix is a fixed DCT-II
/// projection.  Drop-in replacement for `Linear` wherever no bias
/// and no learned weights are needed.
///
/// Forward signature matches `Linear`:
///   `Tensor<N>` -> `Tensor<N>`  (last dim: in_features -> out_features)
#[derive(Module, Debug)]
pub struct DctLinear {
    weight: Tensor<3>,
    
}

#[derive(Config, Debug)]
pub struct DctLinearConfig {
    in_features: usize,
    out_features: usize,
}

impl DctLinearConfig {
    pub fn init(&self, device: &Device) -> DctLinear {
        assert!(self.in_features > 0 && self.out_features > 0);

        let data = build_dct_projection(self.in_features, self.out_features);
        let weight = Tensor::<2>::from_data(
            TensorData::new(data, Shape::new([self.out_features, self.in_features])),
            device,
        )
        .unsqueeze_dim(0)
        .transpose();

        DctLinear { weight }
    }
}

impl DctLinear {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        x.matmul(self.weight.clone())
    }
}
