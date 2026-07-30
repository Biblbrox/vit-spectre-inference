use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::{Distribution, Int, TensorData, backend::Backend},
};
/// Permuter implementation with indicies
///
/// * `signs`: [TODO:parameter]
/// * `perms`: [TODO:parameter]
/// * `num_heads`: [TODO:parameter]
#[derive(Module, Debug)]
pub struct StaticMixer<B: Backend> {
    signs: Tensor<B, 3>,
    perms: Tensor<B, 1, Int>,
    num_heads: usize,
    perm_matrix: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub enum PermutationStrategy {
    Random,
    Xor,
}

#[derive(Config, Debug)]
pub struct StaticMixerConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    strategy: PermutationStrategy,
}

impl<B: Backend> StaticMixer<B> {
    pub fn forward_hard(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let h = self.num_heads;
        // [H * N, E]
        let perms = self.perms.clone();
        let signs = self.signs.clone();

        // [B, N * H, E]
        let x = x.reshape([b, n * h, e / h]);
        let x = x.select(1, perms);

        // [B, N * H, E]
        (x * signs).reshape([b, n, e])
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let h = self.num_heads;

        let x = x.reshape([b, n * h, e / h]); // [B, H*N, E/H]

        // [1, H*N, H*N] @ [B, H*N, E/H] -> [B, H*N, E/H]
        let x = self.perm_matrix.clone().matmul(x);

        (x * self.signs.clone()).reshape([b, n, e])
    }
}

impl StaticMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StaticMixer<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.seq_length;
        let hn = self.num_heads * d;

        let mut sign_per_head = Vec::<Tensor<B, 2>>::new();
        let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
        (0..self.num_heads).for_each(|h| {
            let perm: Tensor<B, 1, Int> = match self.strategy {
                PermutationStrategy::Random => {
                    Tensor::<B, 1>::random([d], Distribution::Uniform(0.0, 1.0), device).argsort(0)
                }
                PermutationStrategy::Xor => Tensor::from_data(
                    TensorData::new((0..d).map(|x| (x ^ (h + 1)) as i32).collect(), [d]),
                    device,
                ),
            };

            perms_per_head.push(perm);
            sign_per_head.push(
                Tensor::<B, 2>::random([d, self.embed_dim / self.num_heads], distribution, device)
                    .sign(),
            )
        });
        let perms = Tensor::cat(perms_per_head, 0);
        let signs = Tensor::cat(sign_per_head, 0).unsqueeze();

        let perm_matrix = Tensor::<B, 2>::zeros([hn, hn], device).scatter(
            1,
            perms.clone().reshape([hn, 1]),
            Tensor::<B, 2>::ones([hn, 1], device),
            burn::tensor::IndexingUpdateOp::Add,
        ); // [H*N, H*N]

        StaticMixer {
            signs,
            perms,
            num_heads: self.num_heads,
            perm_matrix: perm_matrix.unsqueeze_dim(0),
        }
    }
}
