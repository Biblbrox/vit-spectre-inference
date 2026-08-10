use std::sync::Arc;

use burn::{data::dataloader::DataLoader, tensor::Device};
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::{
        batch::{Batch, Batcher},
        dataloader::{
            strategy::{FrameBatchStrategy, fixed::FixedBatchStrategy},
            stream::StreamingDataLoader,
        },
        mapper::LazyMapper,
    },
};

#[cfg(feature = "in_memory_loader")]
use crate::data::dataloader::inmemory::InMemoryDataLoader;

pub struct StreamingDataLoaderBuilder {
    batcher: Arc<dyn Batcher>,
    strategy: Option<Box<dyn FrameBatchStrategy>>,
    mapper: Option<LazyMapper>,
    transforms: Option<Arc<Pipeline>>,
    device: Option<Device>,
    
}

impl StreamingDataLoaderBuilder {
    pub fn new(batcher: Arc<dyn Batcher>) -> Self {
        Self {
            batcher,
            strategy: None,
            mapper: None,
            transforms: None,
            device: None,
        }
    }

    pub fn with_strategy(mut self, strategy: impl FrameBatchStrategy + 'static) -> Self {
        self.strategy = Some(Box::new(strategy));
        self
    }

    pub fn with_mapper(mut self, mapper: LazyMapper) -> Self {
        self.mapper = Some(mapper);
        self
    }

    pub fn with_transforms(mut self, transforms: Arc<Pipeline>) -> Self {
        self.transforms = Some(transforms);
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn build(self, dataset: LazyFrame) -> Arc<dyn DataLoader<Batch>> {
        Arc::new(StreamingDataLoader::new(
            dataset,
            self.batcher,
            self.strategy
                .unwrap_or(Box::new(FixedBatchStrategy::new(1))),
            self.transforms
                .unwrap_or(Arc::new(Pipeline::default())),
            self.device.unwrap_or_default(),
        ))
    }
}

#[cfg(feature = "in_memory_loader")]
pub struct InMemoryDataLoaderBuilder {
    batcher: Arc<dyn Batcher>,
    transforms: Option<Arc<Pipeline>>,
    batch_size: Option<usize>,
    num_workers: Option<usize>,
    device: Option<Device>,
    
}

#[cfg(feature = "in_memory_loader")]
impl InMemoryDataLoaderBuilder {
    pub fn new(batcher: Arc<dyn Batcher>) -> Self {
        Self {
            batcher,
            transforms: None,
            batch_size: None,
            num_workers: None,
            device: None,
        }
    }

    pub fn with_transforms(mut self, transforms: Arc<Pipeline>) -> Self {
        self.transforms = Some(transforms);
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = Some(num_workers);
        self
    }

    pub fn build(self, dataset: LazyFrame) -> Arc<dyn DataLoader<Batch>> {
        Arc::new(InMemoryDataLoader::new(
            dataset,
            self.batcher,
            self.transforms
                .unwrap_or(Arc::new(Pipeline::default())),
            self.batch_size.unwrap_or(1),
            self.num_workers.unwrap_or(0),
            self.device.unwrap_or_default(),
        ))
    }
}
