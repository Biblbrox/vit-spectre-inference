use burn::tensor::Device;
use serde::{Deserialize, Serialize};

use crate::augmentations::colors::{ColorJitter, GaussianBlur, RandomErasing, RandomGrayscale};
use crate::augmentations::normalize::Normalize;
use crate::augmentations::rotation::{Orientation, RandomAffine, RandomFlip};
use crate::augmentations::{Augmentation, Pipeline};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    pub name: String,
    #[serde(flatten)]
    pub params: std::collections::HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AugmentationConfig {
    #[serde(default)]
    pub transforms_train: Vec<TransformConfig>,
    #[serde(default)]
    pub transforms_val: Vec<TransformConfig>,
}

/// A single transform definition from a pipeline defaults section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDefault {
    pub name: String,
    #[serde(default)]
    pub kind: String,
    #[serde(flatten)]
    pub params: std::collections::HashMap<String, toml::Value>,
}

/// A complete pipeline defaults section (e.g., `[defaults.image]`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineDefaults {
    #[serde(default)]
    pub transforms: Vec<PipelineDefault>,
}

pub struct AugmentationBuilder {}

impl Default for AugmentationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AugmentationBuilder {
    pub fn new() -> AugmentationBuilder {
        AugmentationBuilder {}
    }

    fn match_transform(
        &self,
        transform: &TransformConfig,
        transforms_vec: &mut Vec<Box<dyn Augmentation>>,
        device: &Device,
    ) {
        match transform.name.as_str() {
            "normalize" => {
                let mean = transform
                    .params
                    .get("mean")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_float().map(|f| f as f32))
                            .collect::<Vec<f32>>()
                    })
                    .unwrap_or_default();
                let std = transform
                    .params
                    .get("std")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_float().map(|f| f as f32))
                            .collect::<Vec<f32>>()
                    })
                    .unwrap_or_default();
                transforms_vec.push(Box::new(Normalize::new(std, mean, device)));
            }
            "random_flip" => {
                let p = transform.params["probability"].as_float().unwrap();
                let orientation = match transform.params["orientation"].as_str().unwrap() {
                    "horizontal" => Orientation::Horizontal,
                    _ => Orientation::Vertical,
                };
                transforms_vec.push(Box::new(RandomFlip::new(p, orientation)));
            }
            "random_affine" => {
                let p = transform.params["probability"].as_float().unwrap();
                let degrees = transform.params["degrees"].as_float().unwrap();
                transforms_vec.push(Box::new(RandomAffine::new(p, degrees as f32)));
            }
            "color_jitter" => {
                let brightness = transform.params["brightness"].as_float().unwrap();
                let contrast = transform.params["contrast"].as_float().unwrap();
                let saturation = transform.params["saturation"].as_float().unwrap();
                transforms_vec.push(Box::new(ColorJitter::new(
                    brightness as f32,
                    contrast as f32,
                    saturation as f32,
                )));
            }
            "random_erasing" => {
                let p = transform.params["probability"].as_float().unwrap();
                let min_scale = transform.params["min_scale"].as_float().unwrap();
                let max_scale = transform.params["max_scale"].as_float().unwrap();
                let mut er = RandomErasing::new();
                er = er.with_p(p).with_scale(min_scale, max_scale);
                transforms_vec.push(Box::new(er));
            }
            "gaussian_blur" => {
                let p = transform.params["probability"].as_float().unwrap();
                let kernel_size = transform.params["kernel_size"].as_integer().unwrap() as usize;
                let min_sigma = transform.params["min_sigma"].as_float().unwrap();
                let max_sigma = transform.params["max_sigma"].as_float().unwrap();
                let mut gb = GaussianBlur::new(kernel_size, device);
                gb = gb.with_p(p).with_sigma(min_sigma, max_sigma);
                transforms_vec.push(Box::new(gb));
            }
            "random_grayscale" => {
                let p = transform.params["probability"].as_float().unwrap();
                transforms_vec.push(Box::new(RandomGrayscale::new(p)));
            }
            _ => {
                eprintln!("Unknown augmentation: {}", transform.name);
            }
        }
    }

    pub fn build(
        &self,
        config: &AugmentationConfig,
        device: &Device,
        training_device: &Device,
    ) -> (Pipeline, Pipeline) {
        let mut transforms_train: Vec<Box<dyn Augmentation>> = Vec::new();
        let mut transforms_val: Vec<Box<dyn Augmentation>> = Vec::new();

        for transform in &config.transforms_train {
            self.match_transform(transform, &mut transforms_train, training_device);
        }

        for transform in &config.transforms_val {
            self.match_transform(transform, &mut transforms_val, device);
        }

        (
            Pipeline::new(transforms_train),
            Pipeline::new(transforms_val),
        )
    }
}
