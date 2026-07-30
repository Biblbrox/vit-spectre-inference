use burn::{
    Tensor,
    nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    tensor::{Distribution, Int, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use lightmix::attention::{
    learnedmixer::{LearnedPermuter, LearnedPermuterConfig},
    staticmixer::{PermutationStrategy, StaticMixer, StaticMixerConfig},
    stochasticwindowmixer::{StochasticWindowMixer, StochasticWindowMixerConfig},
};

use crate::common::{
    CpuBackend, GpuAutodiffBackend, GpuBackend, generate_run_id, print_bench_results,
};
mod common;

impl<B: Backend> Benchmark for SelfAttentionBenchmark<B> {
    type Input = (Tensor<B, 3>, MultiHeadAttention<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            MultiHeadAttentionConfig::new(self.embed_dim, self.num_heads).init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "SelfAttention-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mha) = input;
        let res = mha.forward(MhaInput::self_attn(tensor));
        Ok(res.context)
    }
}

pub struct SelfAttentionBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub device: B::Device,
}

#[derive(Clone, Copy)]
pub enum StochasticMixerMode {
    Soft,
    Hard,
    Inference,
}

pub struct StochasticMixerBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub mode: StochasticMixerMode,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for StochasticMixerBenchmark<B> {
    type Input = (Tensor<B, 3>, Tensor<B, 4, Int>, StochasticMixer<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        let mixer =
            StochasticMixerConfig::new(self.embed_dim, self.num_heads, 0.01).init(&self.device);
        let (perm_q_base, perm_k_base) = mixer.extract_permutations();

        // [1, H, 1, d] -> [B, H, N, d]
        let perm_q = perm_q_base
            .unsqueeze_dim::<4>(0)
            .repeat_dim(0, self.batch_size)
            .repeat_dim(2, self.num_tokens);
        let perm_k = perm_k_base
            .unsqueeze_dim::<4>(0)
            .repeat_dim(0, self.batch_size)
            .repeat_dim(2, self.num_tokens);

        let perm_qk = Tensor::cat(vec![perm_q, perm_k], 3).expand([
            self.batch_size,
            self.num_heads,
            self.num_tokens,
            2 * self.embed_dim / self.num_heads,
        ]);
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            perm_qk,
            mixer,
        )
    }

    fn name(&self) -> String {
        let mode = match self.mode {
            StochasticMixerMode::Soft => "soft",
            StochasticMixerMode::Hard => "hard",
            StochasticMixerMode::Inference => "inference",
        };
        format!(
            "StochasticMixer[{}]-{:?}x{:?}x{:?} {:?} heads",
            mode, self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, perm_qk, mixer) = input;
        Ok(match self.mode {
            StochasticMixerMode::Soft => mixer.forward(tensor),
            StochasticMixerMode::Hard => mixer.forward_hard(tensor),
            StochasticMixerMode::Inference => mixer.forward_inference(tensor, perm_qk),
        })
    }
}

pub struct StochasticWinMixerBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub mode: StochasticMixerMode,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for StochasticWinMixerBenchmark<B> {
    type Input = (Tensor<B, 3>, StochasticWindowMixer<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        let mixer = StochasticWindowMixerConfig::new(
            self.embed_dim,
            self.num_tokens,
            self.num_heads,
            3,
            0.01,
        )
        .init(&self.device);
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            mixer,
        )
    }

    fn name(&self) -> String {
        let mode = match self.mode {
            StochasticMixerMode::Soft => "soft",
            StochasticMixerMode::Hard => "hard",
            _ => todo!("Not implemented"),
        };
        format!(
            "StochasticWinMixer[{}]-{:?}x{:?}x{:?} {:?} heads",
            mode, self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mixer) = input;
        Ok(match self.mode {
            StochasticMixerMode::Soft => mixer.forward_soft(tensor),
            StochasticMixerMode::Hard => mixer.forward_hard(tensor),
            _ => todo!("Not umplemented"),
        })
    }
}

pub struct LearnedPermutBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for LearnedPermutBenchmark<B> {
    type Input = (Tensor<B, 3>, LearnedPermuter<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            LearnedPermuterConfig::new(self.embed_dim, self.num_tokens, self.num_heads, 0.05)
                .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "LearnedPermuter-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mh_permut) = input;
        let res = mh_permut.forward(tensor);
        Ok(res)
    }
}

pub struct StaticPermuterBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for StaticPermuterBenchmark<B> {
    type Input = (Tensor<B, 3>, StaticMixer<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            StaticMixerConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                PermutationStrategy::Random,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "MHPermutMixMatrix-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mh_permut) = input;
        let res = mh_permut.forward_hard(tensor);
        Ok(res)
    }
}

fn main() {
    println!("=== Mixing Benchmarks ===");
    let run_id = generate_run_id();
    mixing_benchmark_backend::<GpuBackend>(&run_id, "Gpu");
    mixing_benchmark_backend::<CpuBackend>(&run_id, "Cpu");
    mixing_benchmark_backend::<GpuAutodiffBackend>(&run_id, "Gpu (Autodiff)");
}

fn mixing_benchmark_backend<B: Backend>(run_id: &str, backend: &str) {
    use cubecl::benchmark::BenchmarkComputations;
    use cubecl::profile::TimingMethod;

    let device = B::Device::default();

    let batches = [64; 6];
    let embed_dim = [64, 128, 256, 512, 1024, 2048];
    let num_tokens = [64; 6];
    let num_heads = [1, 2, 4, 8, 16];

    let mut results_attn: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_static: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_learned: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_stochastic_soft: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_stochastic_hard: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_stochastic_inference: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_stochastic_win_soft: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_stochastic_win_hard: Vec<(u32, BenchmarkComputations)> = Vec::new();

    for head in num_heads.into_iter() {
        let bench_attn = SelfAttentionBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_static = StaticPermuterBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };
        let bench_learned = LearnedPermutBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_stochastic_soft = StochasticMixerBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            mode: StochasticMixerMode::Soft,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_stochastic_hard = StochasticMixerBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            mode: StochasticMixerMode::Hard,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_stochastic_inference = StochasticMixerBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            mode: StochasticMixerMode::Inference,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_stochastic_win_soft = StochasticWinMixerBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            mode: StochasticMixerMode::Soft,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_stochastic_win_hard = StochasticWinMixerBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            mode: StochasticMixerMode::Hard,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_res_attn = bench_attn.run(TimingMethod::System).unwrap();
        results_attn.push((head as u32, BenchmarkComputations::new(&bench_res_attn)));

        let bench_res_static = bench_static.run(TimingMethod::System).unwrap();
        results_static.push((head as u32, BenchmarkComputations::new(&bench_res_static)));

        let bench_res_learned = bench_learned.run(TimingMethod::System).unwrap();
        results_learned.push((head as u32, BenchmarkComputations::new(&bench_res_learned)));

        let bench_res_stochastic_soft = bench_stochastic_soft.run(TimingMethod::System).unwrap();
        results_stochastic_soft.push((
            head as u32,
            BenchmarkComputations::new(&bench_res_stochastic_soft),
        ));
        let bench_res_stochastic_hard = bench_stochastic_hard.run(TimingMethod::System).unwrap();
        results_stochastic_hard.push((
            head as u32,
            BenchmarkComputations::new(&bench_res_stochastic_hard),
        ));

        let bench_res_stochastic_win_soft =
            bench_stochastic_win_soft.run(TimingMethod::System).unwrap();
        results_stochastic_win_soft.push((
            head as u32,
            BenchmarkComputations::new(&bench_res_stochastic_win_soft),
        ));
        let bench_res_stochastic_win_hard =
            bench_stochastic_win_hard.run(TimingMethod::System).unwrap();
        results_stochastic_win_hard.push((
            head as u32,
            BenchmarkComputations::new(&bench_res_stochastic_win_hard),
        ));

        let bench_res_stochastic_inference = bench_stochastic_inference
            .run(TimingMethod::System)
            .unwrap();
        results_stochastic_inference.push((
            head as u32,
            BenchmarkComputations::new(&bench_res_stochastic_inference),
        ));
    }

    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("SelfAttention/ViT ({})", backend),
        "num_heads",
        &results_attn,
    );
    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("StaticPermut ({})", backend),
        "num_heads",
        &results_static,
    );
    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("LearnedPermut ({})", backend),
        "num_heads",
        &results_learned,
    );
    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("StochasticPermut Soft ({})", backend),
        "num_heads",
        &results_stochastic_soft,
    );
    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("StocasticPermut Hard ({})", backend),
        "num_heads",
        &results_stochastic_hard,
    );
    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("StochasticPermut Inference ({})", backend),
        "num_heads",
        &results_stochastic_inference,
    );

    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("StochasticWinPermut Soft ({})", backend),
        "num_heads",
        &results_stochastic_win_soft,
    );
    print_bench_results(
        run_id,
        "mixing",
        backend,
        &format!("StocasticWinPermut Hard ({})", backend),
        "num_heads",
        &results_stochastic_win_hard,
    );
}
