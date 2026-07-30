use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct DynamicERF<B: Backend> {
    normalized_shape: usize,
    /// Stored flat — reshaped to trailing dim at runtime
    /// There's no other way to support multiple dimensions of tensors... probably
    alpha: Param<Tensor<B, 1>>,
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    shift: Param<Tensor<B, 1>>,
}

#[derive(Config, Debug)]
pub struct DynamicERFConfig {
    normalized_shape: usize,
    #[config(default = 0.5)]
    alpha_init_value: f32,
    #[config(default = 0.0)]
    shift_init_value: f32,
}

impl<B: Backend> DynamicERF<B> {
    /// Works for any rank D >= 1.
    /// Parameters broadcast over all leading dimensions — only the last
    /// dimension must equal `normalized_shape`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // Build a shape [1, 1, ..., 1, normalized_shape] for D dimensions
        let mut param_shape = [1usize; D];
        param_shape[D - 1] = self.normalized_shape;

        // Scalars (alpha, shift): broadcast as all-ones shape except last = 1
        let scalar_shape = [1usize; D];
        // scalar_shape is already all ones — correct for broadcasting
        let weight = self.weight.val().reshape(param_shape);
        let bias = self.bias.val().reshape(param_shape);

        // alpha and shift are scalar-like — stored as length-1 tensors
        let alpha_s = self.alpha.val().reshape(scalar_shape);
        let shift_s = self.shift.val().reshape(scalar_shape);

        let x = alpha_s * x + shift_s;
        x.erf() * weight + bias
    }
}

impl DynamicERFConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DynamicERF<B> {
        let n = self.normalized_shape;
        DynamicERF {
            normalized_shape: n,
            // alpha and shift are scalars — store as length-1 1D tensor
            alpha: Param::from_tensor(Tensor::<B, 1>::ones([1], device) * self.alpha_init_value)
                .set_require_grad(true),
            shift: Param::from_tensor(Tensor::<B, 1>::ones([1], device) * self.shift_init_value)
                .set_require_grad(true),
            // weight and bias are per-feature — store as length-n 1D tensor
            weight: Param::from_tensor(Tensor::<B, 1>::ones([n], device)).set_require_grad(true),
            bias: Param::from_tensor(Tensor::<B, 1>::zeros([n], device)).set_require_grad(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::Tensor;

    use crate::norm::DynamicERFConfig;

    /// Test the erf function in comparison with python implementation from
    /// https://github.com/zlab-princeton/Derf/blob/main/ViT/dynamic_erf.py
    #[test]
    fn test_valid_erf() {
        type B = burn::backend::flex::Flex;
        let device = burn::backend::flex::FlexDevice::default();

        let input = Tensor::<B, 3>::from_floats(
            [
                [
                    [0.5946, 0.7705, 0.0795, 0.2642, 0.3723],
                    [0.1549, 0.1286, 0.0251, 0.8188, 0.8241],
                    [0.5291, 0.2485, 0.4464, 0.2747, 0.7048],
                    [0.6345, 0.9383, 0.5122, 0.3911, 0.3366],
                    [0.9925, 0.2487, 0.6822, 0.0204, 0.4838],
                ],
                [
                    [0.6919, 0.9451, 0.0671, 0.9429, 0.6206],
                    [0.7329, 0.1193, 0.4741, 0.9074, 0.4031],
                    [0.2428, 0.8613, 0.7011, 0.3305, 0.9883],
                    [0.5038, 0.3885, 0.7298, 0.9398, 0.6938],
                    [0.2780, 0.2933, 0.2472, 0.2172, 0.0353],
                ],
                [
                    [0.7738, 0.7314, 0.6581, 0.0363, 0.1245],
                    [0.9519, 0.3632, 0.1753, 0.1310, 0.7091],
                    [0.6445, 0.8702, 0.1613, 0.5437, 0.2420],
                    [0.0196, 0.7139, 0.7264, 0.5712, 0.2673],
                    [0.7292, 0.0209, 0.7893, 0.2614, 0.4873],
                ],
                [
                    [0.1918, 0.5689, 0.6464, 0.6666, 0.6702],
                    [0.0757, 0.4645, 0.9644, 0.0862, 0.3321],
                    [0.9169, 0.0240, 0.2180, 0.6027, 0.1903],
                    [0.5399, 0.7278, 0.9269, 0.1732, 0.4444],
                    [0.2079, 0.1968, 0.5484, 0.2601, 0.5503],
                ],
            ],
            &device,
        );

        let output = Tensor::<B, 3>::from_floats(
            [
                [
                    [0.32584011, 0.41412665, 0.04482946, 0.14819636, 0.20764662],
                    [0.08721854, 0.07245491, 0.01416042, 0.43739668, 0.4399227],
                    [0.29169255, 0.13948296, 0.24773369, 0.15401378, 0.38177592],
                    [0.34632252, 0.49297572, 0.28278255, 0.21787393, 0.18812832],
                    [0.51719827, 0.13959407, 0.37046983, 0.01150907, 0.26772305],
                ],
                [
                    [0.37533329, 0.49604935, 0.03784292, 0.49505602, 0.33921562],
                    [0.39570817, 0.06722807, 0.26255544, 0.4788857, 0.22438248],
                    [0.13631523, 0.45749616, 0.379931, 0.18478117, 0.51534399],
                    [0.27833946, 0.21646171, 0.39417809, 0.49365457, 0.37628402],
                    [0.1558404, 0.16429816, 0.13876069, 0.12206193, 0.01991382],
                ],
                [
                    [0.41573066, 0.39496802, 0.358317, 0.02047783, 0.07015098],
                    [0.49911311, 0.20268318, 0.09864974, 0.07380328, 0.38391699],
                    [0.35141608, 0.46165944, 0.09080686, 0.29935798, 0.13587046],
                    [0.01105776, 0.38630316, 0.39249795, 0.3137133, 0.14991474],
                    [0.39388174, 0.01179113, 0.42323713, 0.14664367, 0.26958469],
                ],
                [
                    [0.10788074, 0.31251691, 0.35238202, 0.36261445, 0.3644309],
                    [0.04268876, 0.25742939, 0.50471918, 0.04860305, 0.18565944],
                    [0.48323895, 0.0135399, 0.12250797, 0.3300184, 0.10704214],
                    [0.29736577, 0.39319002, 0.48780087, 0.0974739, 0.24665991],
                    [0.1168739, 0.11067519, 0.3018192, 0.14592259, 0.30281326],
                ],
            ],
            &device,
        );

        let erf = DynamicERFConfig::new(5).init(&device);
        let x = erf.forward(input);
        let equal = x.clone().all_close(output.clone(), Some(1e-5), Some(1e-5));
        if !equal {
            println!("output: {:?}", x.to_string());
            println!("expected: {:?}", output.to_string());
        }
        assert!(equal);
    }
}
