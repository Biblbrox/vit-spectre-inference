#![recursion_limit = "2048"]

use std::{fs::File, io::Write, panic, path::PathBuf};

use burn::tensor::backend::Backend;
use lightmix::models::efficientvit::EfficientViTConfig;
use lightmix::models::fast_vit::FastViTConfig;
use lightmix::models::fast_vit3d::FastViT3DConfig;
use lightmix::models::vit::ViTConfig;
use lightmix::utils::print_model_info;
use lightmix::{config::ParsedConfig, training::run_experiment};

#[cfg(feature = "jemalloc")]
use tikv_jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

macro_rules! info_for_model {
    (
        $model_name:expr,
        $model_table:expr,
        $shared:expr,
        $dataset_cfg:expr,
        $device:expr,
        $( $prefix:literal => $config_type:ty ),* $(,)?
    ) => {
        match $model_name.as_str() {
            $(
                name if name.starts_with($prefix) => {
                    let model_cfg: $config_type = $model_table.try_into().unwrap();
                    print_model_info::<B>($shared, $dataset_cfg, $device, model_cfg);
                }
            )*
            _ => panic!("Unknown model: {}", $model_name),
        }
    };
}

pub fn run_info<B: Backend>(config: ParsedConfig, device: B::Device) {
    let ParsedConfig {
        shared,
        dataset: dataset_cfg,
        model_table,
    } = config;
    let model_name = shared.active_model.clone();

    info_for_model!(
        model_name,
        model_table,
        shared,
        dataset_cfg,
        device,
        "fast_vit_cloud" => FastViT3DConfig,
        "fast_vit"      => FastViTConfig,
        "vit"           => ViTConfig,
        "efficientvit"  => EfficientViTConfig,
    );
}

fn main() {
    type MyBackend = burn::backend::cuda::Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::default();

    let config_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let localpath = config_dir.join("experiments.local.toml");

    let config = ParsedConfig::load(&config_dir, Some(&localpath));
    println!("Config loaded from {}", config_dir.display());

    panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::capture();

        let msg = format!("=== PANIC ===\n{info}\n\n=== BACKTRACE ===\n{backtrace}\n",);

        if let Ok(mut f) = File::create("panic.log") {
            let _ = f.write_all(msg.as_bytes());
        }

        eprintln!("{msg}");
    }));

    let command = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "train".to_string());
    match command.as_str() {
        "info" => run_info::<MyBackend>(config, device),
        "train" => run_experiment::<MyBackend>(config, device),
        other => eprintln!("Unknown command: '{other}' (expected 'train' or 'info')"),
    }
}
