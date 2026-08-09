use std::{
    panic,
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{
    data::dataloader::Progress,
    grad_clipping::GradientClippingConfig,
    lr_scheduler::{
        cosine::CosineAnnealingLrSchedulerConfig, module_lr_scheduler::ModuleLearningRate,
    },
    module::{AutodiffModule, Module},
    optim::AdamWConfig,
    tensor::Device,
    train::{
        ClassificationOutput, InferenceStep, Interrupter, LearnerSummary, TrainStep,
        logger::{FileMetricLogger, MetricLogger, TrainingProgressLogger},
        metric::{
            AccuracyMetric, LossMetric, MetricMetadata, TopKAccuracyMetric,
            store::{EpochSummary, Split},
        },
        renderer::tui::TuiMetricsRendererWrapper,
    },
};
use polars::prelude::PlRefPath;
use serde::Serialize;

use crate::{
    augmentations::{Pipeline, builder::AugmentationBuilder},
    config::ParsedConfig,
    data::{
        batch::Batch,
        dataset::{DatasetType, LazyDataset, LazyFiletype},
    },
    metrics::{MetricOutput, MetricsHandler},
    models::{
        ModelConfig, efficientvit::EfficientViTConfig, fast_vit::FastViTConfig,
        fast_vit3d::FastViT3DConfig, vit::ViTConfig,
    },
};
use crate::{config::DatasetConfig, data::dataloader::strategy::buffered::BufferedBatchStrategy};
use crate::{config::SharedConfig, data::builder::StreamingDataLoaderBuilder, models::TrainConfig};

// Training options to customize behavior (artifact dir, TUI, callbacks).
pub struct TrainOptions {
    /// Override artifact directory path. None uses default `./experiments/{model}-{dataset}`.
    pub artifact_dir: Option<String>,
    /// Enable TUI metrics renderer. Default true.
    pub enable_tui: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            artifact_dir: None,
            enable_tui: true,
        }
    }
}

fn save_config<T: Serialize>(value: &T, path: &Path) {
    let content = toml::to_string_pretty(value).expect("serialize config");
    std::fs::write(path, content).expect("write config");
}

/// Build the metric set, pairing each metric with a closure that extracts
/// its specific `Input` type from the type-erased model output.
pub fn build_metrics() -> MetricsHandler {
    use burn::train::metric::{AccuracyInput, LossInput, TopKAccuracyInput};

    MetricsHandler::new()
        .add(LossMetric::new(), |o: &dyn MetricOutput| {
            LossInput::new(o.loss())
        })
        .add(AccuracyMetric::new(), |o: &dyn MetricOutput| {
            AccuracyInput::new(o.output(), o.targets())
        })
        .add(TopKAccuracyMetric::new(5), |o: &dyn MetricOutput| {
            TopKAccuracyInput::new(o.output(), o.targets())
        })
    //.add(BatchTimeMetric::new())
    //.add(ThroughputMetric::new())
}

#[derive(serde::Deserialize)]
struct OptimizerConfig {
    adam_weight_decay: f64,
    adam_betas: [f64; 2],
}

fn match_dataset(dataset_name: &str) -> DatasetType {
    dataset_name
        .parse::<DatasetType>()
        .expect("Unknown dataset")
}

macro_rules! train_for_model {
    (
        $model_name:expr,
        $model_table:expr,
        $dataset_path:expr,
        $shared:expr,
        $dataset_cfg:expr,
        $device:expr,
        $training_device:expr,
        $ds_type:expr,
        $optimizer:expr,
        $options:expr,
        $( $prefix:literal => $config_type:ty ),* $(,)?
    ) => {
        match $model_name.as_str() {
            $(
                name if name.starts_with($prefix) => {
                    let model_cfg: $config_type = $model_table.try_into().unwrap();
                    train(
                        $dataset_path.into(),
                        $shared,
                        $dataset_cfg,
                        $device,
                        $training_device,
                        model_cfg,
                        $ds_type,
                        $optimizer,
                        $options,
                    );
                }
            )*
            _ => panic!("Unknown model: {}", $model_name),
        }
    };
}

pub fn run_experiment(config: ParsedConfig, device: Device, training_device: Device) {
    let optimizer_cfg: OptimizerConfig = config.model_table.clone().try_into().unwrap();
    let ParsedConfig {
        shared,
        dataset: dataset_cfg,
        model_table,
    } = config;

    let dataset_name = shared.active_dataset.clone();
    let model_name = shared.active_model.clone();

    let dataset_path = PathBuf::from(&shared.cache_dir).join(&dataset_name);
    if !dataset_path.exists() {
        panic!("Dataset path {} doesn't exist", dataset_path.display());
    }
    let dataset_path = dataset_path.to_str().unwrap();

    let optimizer = AdamWConfig::new()
        .with_weight_decay(optimizer_cfg.adam_weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .with_beta_1(optimizer_cfg.adam_betas[0] as f32)
        .with_beta_2(optimizer_cfg.adam_betas[1] as f32);

    let ds_type = match_dataset(&dataset_name);

    let options = TrainOptions::default();
    train_for_model!(
        model_name,
        model_table,
        dataset_path,
        shared,
        dataset_cfg,
        device,
        training_device,
        ds_type,
        optimizer,
        options,
        "fast_vit_cloud" => FastViT3DConfig,
        "fast_vit"      => FastViTConfig,
        "vit"           => ViTConfig,
        "efficientvit"  => EfficientViTConfig,
    );
}

fn step_metadata(iteration: usize, total_iterations: usize, lr: f64) -> MetricMetadata {
    let progress = Progress {
        items_processed: iteration + 1,
        items_total: total_iterations,
        unit: Some("batch".to_string()),
    };
    MetricMetadata {
        progress,
        iteration: Some(iteration),
        lr: Some(ModuleLearningRate::from(lr)),
    }
}

pub fn train<M>(
    dataset_path: PlRefPath,
    shared: SharedConfig,
    dataset_cfg: DatasetConfig,
    device: Device,
    training_device: Device,
    model: M,
    dataset: DatasetType,
    optimizer: AdamWConfig,
    options: TrainOptions,
) where
    M: ModelConfig,
    <M as ModelConfig>::TrainModel: InferenceStep<Input = Batch, Output = ClassificationOutput>,
{
    let file_type = dataset_cfg
        .dataset_type
        .parse::<LazyFiletype>()
        .expect("invalid dataset_type");

    let artifact_dir: PathBuf = options
        .artifact_dir
        .unwrap_or_else(|| {
            format!(
                "./experiments/{}-{}",
                shared.active_model, shared.active_dataset
            )
        })
        .into();

    // Remove existing artifacts before to get an accurate learner summary
    if !shared.continue_training {
        std::fs::remove_dir_all(artifact_dir.clone()).ok();
        std::fs::create_dir_all(artifact_dir.clone()).ok();
    }
    let batcher_train = dataset.make_batcher();
    let batcher_val = dataset.make_batcher();
    let dataset = dataset.make_dataset();

    let (pipeline_train, pipeline_val): (Pipeline, Pipeline) =
        AugmentationBuilder::new().build(&dataset_cfg.augmentations, &device, &training_device);

    save_config(&shared, &artifact_dir.join("shared_config.json"));
    save_config(&dataset_cfg, &artifact_dir.join("dataset_config.json"));

    // Seed the random number generator
    // Note: Autodiff::seed requires knowing the backend type at compile time
    // For now, we'll skip this since device is generic

    let strategy_train = BufferedBatchStrategy::new(
        dataset_cfg.batch_size,
        dataset_cfg.batch_size,
        shared.num_workers as usize,
    );

    let dataloader_train = StreamingDataLoaderBuilder::new(batcher_train)
        .with_strategy(strategy_train.with_shuffle(shared.random_seed as u64))
        .with_transforms(Arc::new(pipeline_train))
        .with_device(training_device.clone())
        .build(dataset.train(dataset_path.clone(), file_type.clone()));
    let strategy_val = BufferedBatchStrategy::new(
        dataset_cfg.val_batch_size,
        dataset_cfg.val_batch_size,
        shared.num_workers as usize,
    );
    let dataloader_val = StreamingDataLoaderBuilder::new(batcher_val)
        .with_strategy(strategy_val)
        .with_transforms(Arc::new(pipeline_val))
        .with_device(device.clone())
        .build(dataset.validation(dataset_path, file_type.clone()));

    let num_iterations = dataloader_train.num_items() / dataset_cfg.batch_size;
    let train_config = TrainConfig {
        in_channels: dataset_cfg.in_channels,
        image_size: dataset_cfg.img_size,
        num_classes: dataset_cfg.num_classes,
    };
    let mut model = model.init_training(&training_device, &train_config);
    let mut optimizer = optimizer.init();
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(
        shared.learning_rate,
        dataset_cfg.epochs * num_iterations,
    )
    .init()
    .unwrap();

    let mut train_metrics = build_metrics();
    let mut valid_metrics = build_metrics();

    let mut stop_flag = false;
    let mut logger = FileMetricLogger::new(&artifact_dir);
    for definition in train_metrics.definitions() {
        logger.log_metric_definition(definition.clone());
    }

    let interrupter = Interrupter::new();
    let mut renderer = Box::new(TuiMetricsRendererWrapper::new(interrupter.clone(), None));
    valid_metrics.register(&mut *renderer);
    train_metrics.register(&mut *renderer);

    renderer.start(dataset_cfg.epochs, 1, None);

    for epoch in 1..=dataset_cfg.epochs {
        let mut lr_module = ModuleLearningRate::from(0.0_f64);

        renderer.start_split("train", num_iterations);
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let batch = batch.expect("Failed to load batch");
            if interrupter.should_stop() {
                stop_flag = true;
                break;
            }

            let metadata = step_metadata(iteration, num_iterations, lr_module.base());

            let output = TrainStep::step(&model, batch);
            let item = output.item;
            let grads = output.grads;

            lr_module = scheduler.step();
            model = optimizer.step(lr_module.clone(), model, grads);

            train_metrics.update(
                &item,
                &metadata,
                epoch,
                &mut *renderer,
                &mut logger,
                Split::Train,
            );
            renderer.update_split(iteration + 1);
        }
        renderer.end_split();

        let model_valid = model.valid();
        let num_val_iterations = dataloader_val.num_items() / dataset_cfg.val_batch_size;

        renderer.start_split("valid", num_val_iterations);
        for (iteration, batch) in dataloader_val.iter().enumerate() {
            let batch = batch.expect("Failed to load validation batch");
            if interrupter.should_stop() {
                stop_flag = true;
                break;
            }

            let metadata = step_metadata(iteration, num_val_iterations, lr_module.base());

            let output = InferenceStep::step(&model_valid, batch);

            valid_metrics.update(
                &output,
                &metadata,
                epoch,
                &mut *renderer,
                &mut logger,
                Split::Valid,
            );
            renderer.update_split(iteration + 1);
        }
        renderer.end_split();

        renderer.update_epoch(epoch);

        let model_path = artifact_dir.join(format!("model-epoch-{epoch}"));
        let optim_path = artifact_dir.join(format!("optim-epoch-{epoch}"));
        let sched_path = artifact_dir.join(format!("scheduler-epoch-{epoch}"));
        model
            .clone()
            .save_file(&model_path)
            .expect("Checkpoint save failed");

        optimizer.clone().to_record().save(&optim_path).ok();
        scheduler.clone().to_record().save(&sched_path).ok();

        logger.log_epoch_summary(EpochSummary {
            epoch_number: epoch,
            split: Split::Train,
        });

        if stop_flag {
            interrupter.stop(Some("Training finished"));
            break;
        }
    }

    renderer.end();

    // If we don't do that, renderer won't allow stdout to pass
    drop(*renderer);

    let metric_names = train_metrics.metric_names();
    println!("{}", model);
    match LearnerSummary::new(artifact_dir, &metric_names) {
        Ok(summary) => eprintln!("{}", summary),
        Err(e) => eprintln!("Summary unavailable: {}", e),
    }
}
