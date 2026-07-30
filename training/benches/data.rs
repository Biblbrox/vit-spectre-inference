use std::sync::Arc;
use std::time::Instant;

use indicatif::ProgressBar;
use lightmix::augmentations::colors::ColorJitter;
use lightmix::augmentations::normalize::Normalize;
use lightmix::augmentations::rotation::RandomAffine;
use lightmix::augmentations::{Augmentation, Pipeline};
use lightmix::data::batch::Cifar100Batcher;
use lightmix::data::batch::ImageNet1kBatcher;
use lightmix::data::builder::StreamingDataLoaderBuilder;
use lightmix::data::dataloader::strategy::buffered::BufferedBatchStrategy;
use lightmix::data::dataset::Cifar100Dataset;
use lightmix::data::dataset::ImageNet1kDataset;
use lightmix::data::dataset::{LazyDataset, LazyFiletype};
use polars::prelude::PlRefPath;

use crate::common::{CpuBackend, CpuDevice};

mod common;

fn main() {
    println!("=== ImageNet1k Dataset Benchmark ===");
    test_imagenet1k();

    println!("\n=== CIFAR100 Dataset Benchmark ===");
    test_cifar100();
}

fn test_imagenet1k() {
    // Substitude your path to load dataset
    // TODO: I should move it to the config
    let imagenet1k_path: PlRefPath = "/storage/experiments-ml/datasets/imagenet1k".into();

    let shuffle_seed = 42;
    let batch_size = 128;

    type B = CpuBackend;
    let device = CpuDevice::default();

    let ds = ImageNet1kDataset {};

    let std = vec![0.229, 0.224, 0.225];
    let mean = vec![0.485, 0.456, 0.406];

    let normalize = Box::new(Normalize::<B>::new(std, mean, &device));
    let random_rotate = Box::new(RandomAffine::<B>::new(0.5, 30.0));
    let color_jitter = Box::new(ColorJitter::<B>::new(0.4, 0.4, 0.1));

    let transforms: Vec<Box<dyn Augmentation<B>>> = vec![normalize, color_jitter, random_rotate];
    let pipeline = Pipeline::new(transforms);

    let batcher = ImageNet1kBatcher::new();
    let strategy = BufferedBatchStrategy::new(batch_size, 10, 4); //.with_mapper(Mapper::decoder());
    let dl = StreamingDataLoaderBuilder::<B>::new(batcher.clone())
        .with_strategy(strategy.clone().with_shuffle(shuffle_seed))
        .with_transforms(Arc::new(pipeline))
        .with_device(device)
        .build(ds.test(imagenet1k_path, LazyFiletype::Arrow));

    let pbar = ProgressBar::new(dl.num_items() as u64);
    let start = Instant::now();
    for _df in dl.iter() {
        pbar.inc(batch_size as u64);
    }
    let duration = start.elapsed();
    pbar.finish_with_message("Done");
    println!("ImageNet1k train dataset preparing time: {:?}", duration);
}

fn test_cifar100() {
    let cifar100_path: PlRefPath = "/storage/experiments-ml/datasets/cifar100".into();

    let shuffle_seed = 42;
    let batch_size = 128;

    type B = CpuBackend;
    let device = CpuDevice::default();

    let ds = Cifar100Dataset {};
    let batcher = Cifar100Batcher::new();
    let strategy = BufferedBatchStrategy::new(batch_size, 10, 4);

    let dl = StreamingDataLoaderBuilder::<B>::new(batcher.clone())
        .with_strategy(strategy.clone().with_shuffle(shuffle_seed))
        .with_device(device)
        .build(ds.train(cifar100_path, LazyFiletype::Arrow));

    let pbar = ProgressBar::new(dl.num_items() as u64);
    let start = Instant::now();
    for _df in dl.iter() {
        pbar.inc(batch_size as u64);
    }
    let duration = start.elapsed();
    pbar.finish_with_message("Done");
    println!("CIFAR100 train dataset preparing time: {:?}", duration);
}
