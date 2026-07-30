use crate::data::batch::Batcher;
use paste::paste;

use super::{LazyDataset, LazyFiletype};
use crate::augmentations::Pipeline;
use crate::data::batch::Batch;
use burn::tensor::Shape;
use burn::tensor::backend::Backend;
use polars::prelude::*;
use std::{str::FromStr, sync::Arc};

pub enum DatasetType {
    Cifar10,
    Cifar100,
    Mnist,
    FashionMnist,
    Food101,
    TinyImageNet,
    ImageNet1k,
    //ModelNet40,
    ImageNette2,
}

impl FromStr for DatasetType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cifar10" => Ok(DatasetType::Cifar10),
            "cifar100" => Ok(DatasetType::Cifar100),
            "mnist" => Ok(DatasetType::Mnist),
            "fashionmnist" => Ok(DatasetType::FashionMnist),
            "food101" => Ok(DatasetType::Food101),
            "tinyimagenet" => Ok(DatasetType::TinyImageNet),
            "imagenet1k" => Ok(DatasetType::ImageNet1k),
            "imagenette2" => Ok(DatasetType::ImageNette2),
            //"modelnet40" => Ok(DatasetType::ModelNet40),
            _ => Err(format!("Unknown dataset: {}", s)),
        }
    }
}

impl DatasetType {
    pub fn make_dataset(&self) -> DynDataset {
        match self {
            Self::Cifar10 => DynDataset(Box::new(Cifar10Dataset {})),
            Self::Cifar100 => DynDataset(Box::new(Cifar100Dataset {})),
            Self::Mnist => DynDataset(Box::new(MnistDataset {})),
            Self::FashionMnist => DynDataset(Box::new(FashionMnistDataset {})),
            Self::Food101 => DynDataset(Box::new(Food101Dataset {})),
            Self::TinyImageNet => DynDataset(Box::new(TinyImageNetDataset {})),
            Self::ImageNet1k => DynDataset(Box::new(ImageNet1kDataset {})),
            Self::ImageNette2 => DynDataset(Box::new(ImageNette2Dataset {})),
            //Self::ModelNet40 => DynDataset(Box::new(ModelNet40Dataset {})),
        }
    }

    pub fn make_batcher<B: Backend>(&self) -> Arc<dyn Batcher<B>> {
        match self {
            Self::Cifar10 => Cifar10Batcher::new(),
            Self::Cifar100 => Cifar100Batcher::new(),
            Self::Mnist => MnistBatcher::new(),
            Self::FashionMnist => FashionMnistBatcher::new(),
            Self::Food101 => Food101Batcher::new(),
            Self::TinyImageNet => TinyImageNetBatcher::new(),
            Self::ImageNet1k => ImageNet1kBatcher::new(),
            Self::ImageNette2 => ImageNet1kBatcher::new(),
            //Self::ModelNet40 => ModelNet40Batcher::new(),
        }
    }
}

pub struct DynDataset(pub Box<dyn LazyDataset>);

impl LazyDataset for DynDataset {
    fn scan(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.scan(path, ft)
    }

    fn train(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.train(uri, ft)
    }

    fn test(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.test(uri, ft)
    }

    fn validation(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.validation(uri, ft)
    }
}

macro_rules! define_dataset {
    ($name:ident, $train_glob:expr, $test_glob:expr, $val_glob:expr, $width:expr, $height:expr, $channels:expr, $data_col:expr, $label_col:expr) => {
        paste! {
            pub struct [<$name Dataset>];

            impl LazyDataset for [<$name Dataset>] {
                fn validation(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
                    let path = path.join($val_glob);
                    self.scan(path, ft)
                }

                fn train(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
                    let path = uri.join($train_glob);
                    self.scan(path, ft)
                }

                fn test(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
                    let path = uri.join($test_glob);
                    self.scan(path, ft)
                }
            }


            pub struct [<$name Batcher>];

            impl [<$name Batcher>] {
                pub fn new() -> Arc<Self> {
                    Arc::new(Self)
                }
            }

            impl<B: Backend> Batcher<B> for [<$name Batcher>] {
                fn batch(
                    &self,
                    df: DataFrame,
                    transforms: Arc<Pipeline<B>>,
                    device: &B::Device,
                ) -> Batch<B> {
                    let b = df.height();
                    self.generic_batch(
                        df,
                        transforms,
                        Shape::new([b, $width, $height, $channels]),
                        $data_col,
                        $label_col,
                        device,
                    )
                }
            }
        }
    };
}

define_dataset!(
    Mnist,
    "**/train*.*",
    "**/test*.*",
    "**/test*.*",
    28,
    28,
    1,
    "image",
    "label"
);
define_dataset!(
    FashionMnist,
    "**/train*.*",
    "**/test*.*",
    "**/test*.*",
    28,
    28,
    1,
    "image",
    "label"
);
define_dataset!(
    Cifar10,
    "**/train*.*",
    "**/test*.*",
    "**/test*.*",
    32,
    32,
    3,
    "image",
    "label"
);
define_dataset!(
    Cifar100,
    "**/train*.*",
    "**/test*.*",
    "**/test*.*",
    32,
    32,
    3,
    "image",
    "label"
);
define_dataset!(
    Food101,
    "**/train*.*",
    "**/val*.*",
    "**/val*.*",
    96,
    96,
    3,
    "image",
    "label"
);

define_dataset!(
    TinyImageNet,
    "**/train*.*",
    "**/test*.*",
    "**/val*.*",
    64,
    64,
    3,
    "image",
    "label"
);
define_dataset!(
    ImageNet1k,
    "**/train*.*",
    "**/test*.*",
    "**/val*.*",
    224,
    224,
    3,
    "image",
    "label"
);
define_dataset!(
    ImageNette2,
    "**/train*.*",
    "**/test*.*",
    "**/test*.*",
    320,
    320,
    3,
    "image",
    "label"
);
//define_dataset!(ModelNet40Dataset, "**/train*.*", "**/test*.*", "**/test*.*");
//define_dataset!(CocoSegDataset, "**/train*.*", "**/test*.*", "**/test*.*");
