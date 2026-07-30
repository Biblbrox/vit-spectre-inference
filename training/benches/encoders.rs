use burn::{
    Tensor,
    nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    tensor::{Distribution, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use lightmix::encoders::fast_encoder::{FastEncoder, FastEncoderConfig};

use cubecl::{benchmark::BenchmarkComputations, profile::TimingMethod};

use crate::common::{CpuBackend, GpuBackend, generate_run_id, print_bench_results};
mod common;

pub struct FastEncoderBenchmark<B: Backend> {
    pub seq_length: usize,
    pub batch_size: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub hid_dim: usize,
    pub dropout: f64,
    pub nhead: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for FastEncoderBenchmark<B> {
    type Input = (Tensor<B, 3>, FastEncoder<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.seq_length, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            FastEncoderConfig::new(
                self.num_layers,
                self.seq_length,
                self.embed_dim,
                self.hid_dim,
                self.dropout,
                self.nhead,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "FastEncoder-{:?}x{:?}x{:?} {:?}",
            self.batch_size, self.seq_length, self.embed_dim, self.nhead
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, model) = input;
        let res = model.forward(tensor);
        Ok(res)
    }
}

pub struct ViTEncoderBenchmark<B: Backend> {
    pub num_patches: usize,
    pub batch_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hid_dim: usize,
    pub dropout: f64,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for ViTEncoderBenchmark<B> {
    type Input = (Tensor<B, 3>, TransformerEncoder<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_patches, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            TransformerEncoderConfig::new(
                self.embed_dim,
                self.hid_dim,
                self.num_heads,
                self.num_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .with_activation(burn::nn::activation::ActivationConfig::Gelu)
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "ViTEncoder-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_patches, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, model) = input;
        let res = model.forward(TransformerEncoderInput::new(tensor));
        Ok(res)
    }
}

fn encoders_benchmark_backend<B: Backend>(run_id: &str, backend: &str) {
    let device = B::Device::default();

    let batch_size = 8;
    let embed_dim = 64;
    let seq_length = 196;
    let num_heads = 4;
    let num_layers = 12;
    let hid_dim = 768;
    let dropout = 0.5;

    let mut results_vit: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut variant_results: Vec<(&str, Vec<(u32, BenchmarkComputations)>)> = Vec::new();

    for (idx, variant) in AttentionVariant::all_variants().iter().enumerate() {
        let mix_layer = make_attention_config(*variant, embed_dim, num_heads, seq_length);

        let bench_fast = FastEncoderBenchmark::<B> {
            seq_length,
            batch_size,
            embed_dim,
            num_layers,
            hid_dim,
            nhead: num_heads,
            device: device.clone(),
            dropout,
        };

        let bench_res_fast = bench_fast.run(TimingMethod::System).unwrap();
        let computed_fast = BenchmarkComputations::new(&bench_res_fast);

        variant_results.push((variant.label(), vec![(idx as u32, computed_fast)]));
    }

    // ViT baseline
    let bench_vit = ViTEncoderBenchmark::<B> {
        num_patches: seq_length,
        batch_size,
        embed_dim,
        num_heads,
        num_layers,
        hid_dim,
        device: device.clone(),
        dropout,
    };
    let bench_res_vit = bench_vit.run(TimingMethod::System).unwrap();
    results_vit.push((0, BenchmarkComputations::new(&bench_res_vit)));

    print_bench_results(
        run_id,
        "encoders",
        backend,
        &format!("ViTEncoder ({})", backend),
        "baseline",
        &results_vit,
    );

    for (label, results) in variant_results {
        print_bench_results(
            run_id,
            "encoders",
            backend,
            &format!("FastEncoder[{}] ({})", label, backend),
            "variant",
            &results,
        );
    }
}

fn main() {
    let run_id = generate_run_id();
    encoders_benchmark_backend::<GpuBackend>(&run_id, "GPU");
    encoders_benchmark_backend::<CpuBackend>(&run_id, "CPU");
}
