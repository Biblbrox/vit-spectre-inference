use burn::{module::Module, tensor::backend::Backend};
use fastrand::Rng;

use crate::{
    config::{DatasetConfig, SharedConfig},
    models::{ModelConfig, TrainConfig},
};

pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Sample from standard normal distribution using Box-Muller transform.
fn sample_normal(rng: &mut Rng) -> f64 {
    loop {
        let u1 = rng.f64();
        let u2 = rng.f64();
        if u1 > 1e-10 {
            return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
}

/// Sample from Gamma(α, 1) distribution using Marsaglia-Tsang method.
/// For α >= 1: Marsaglia-Tsang
/// For 0 < α < 1: use relation Gamma(α) = Gamma(α+1) * U^(1/α)
pub fn sample_gamma(rng: &mut Rng, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    if alpha < 1.0 {
        let u = rng.f64();
        return sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }

    // Marsaglia-Tsang method for α >= 1
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        if x < 0.0 {
            return d * v;
        }
        let u = rng.f64();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if (u + u).ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample from Beta(α, β) distribution using Gamma ratios.
/// If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1), then X/(X+Y) ~ Beta(α, β).
pub fn sample_beta(rng: &mut Rng, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        return 0.5;
    }
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    if x + y == 0.0 {
        return 0.5;
    }
    x / (x + y)
}

pub fn print_model_info<B: Backend>(
    shared: SharedConfig,
    dataset_cfg: DatasetConfig,
    device: B::Device,
    model: impl ModelConfig<B>,
) {
    let train_config = TrainConfig {
        in_channels: dataset_cfg.in_channels,
        image_size: dataset_cfg.img_size,
        num_classes: dataset_cfg.num_classes,
    };

    let model = model.init_training(&device, &train_config);

    println!("=== Configuration ===");
    println!("Model:      {}", shared.active_model);
    println!("Dataset:    {}", shared.active_dataset);
    println!("Image size: {}", dataset_cfg.img_size);
    println!("Classes:    {}", dataset_cfg.num_classes);
    println!();

    println!("=== Augmentations ===");
    let mut aug_table = toml::value::Table::new();
    aug_table.insert(
        "augmentations".to_string(),
        toml::Value::try_from(&dataset_cfg.augmentations).expect("serialize augmentations"),
    );
    println!(
        "{}",
        toml::to_string_pretty(&aug_table).expect("format augmentations")
    );

    println!("=== Model structure ===");
    println!("{}", model);

    let num_params = model.num_params();
    let bytes_f32 = num_params * 4;
    let bytes_f16 = num_params * 2;

    println!("=== Model size ===");
    println!("Parameters: {}", format_param_count(num_params));
    println!("Size (f32): {}", format_bytes(bytes_f32));
    println!("Size (f16): {}", format_bytes(bytes_f16));
}

fn format_param_count(n: usize) -> String {
    match n {
        n if n >= 1_000_000_000 => format!("{:.2}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.2}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.2}K", n as f64 / 1e3),
        n => n.to_string(),
    }
}

fn format_bytes(bytes: usize) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else {
        format!("{:.2} MiB", b / MIB)
    }
}
