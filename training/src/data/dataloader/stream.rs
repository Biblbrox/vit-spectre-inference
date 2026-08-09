use std::{num::NonZero, sync::Arc};

use burn::{
    data::dataset::DatasetError,
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    backend::Backend,
    tensor::Device,
};
use polars::lazy::{
    dsl::{Engine, len},
    frame::LazyFrame,
};

use crate::{
    augmentations::Pipeline,
    data::{
        batch::{Batch, Batcher},
        dataloader::strategy::FrameBatchStrategy,
    },
};

pub struct StreamingDataLoader {
    dataset: LazyFrame,
    batcher: Arc<dyn Batcher>,
    strategy: Box<dyn FrameBatchStrategy>,
    transforms: Arc<Pipeline>,
    total_items: usize,
    device: Device,
    
}

impl Clone for StreamingDataLoader {
    fn clone(&self) -> Self {
        Self {
            dataset: self.dataset.clone(),
            batcher: self.batcher.clone(),
            strategy: self.strategy.clone_dyn(),
            transforms: self.transforms.clone(),
            total_items: self.total_items,
            device: self.device.clone(),
        }
    }
}

impl StreamingDataLoader {
    pub fn new(
        dataset: impl Into<LazyFrame>,
        batcher: Arc<dyn Batcher>,
        strategy: Box<dyn FrameBatchStrategy>,
        transforms: Arc<Pipeline>,
        device: Device,
    ) -> Self {
        let dataset = dataset.into();
        let total_items = dataset
            .clone()
            .select([len()])
            .collect_with_engine(Engine::Streaming)
            .unwrap()
            .unwrap_single()
            .column("len")
            .unwrap()
            .u32()
            .unwrap()
            .get(0)
            .unwrap() as usize;
        Self {
            dataset,
            batcher,
            strategy,
            transforms,
            total_items,
            device,
        }
    }
}

impl DataLoader<Batch> for StreamingDataLoader
where
    
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<Batch> + 'a> {
        Box::new(StreamingDataLoaderIterator::new(
            self.dataset.clone(),
            self.batcher.clone(),
            self.strategy.clone_dyn(),
            self.transforms.clone(),
            self.total_items,
            self.device.clone(),
        ))
    }

    fn num_items(&self) -> usize {
        self.total_items
    }

    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader<Batch>> {
        let mut loader = self.clone();
        loader.device = device.clone();
        Arc::new(loader)
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<Batch>> {
        let mut loader = self.clone();
        loader.dataset = self
            .dataset
            .clone()
            .slice(start as i64, (end - start) as u32);
        loader.total_items = end - start;
        Arc::new(loader)
    }
}

pub struct StreamingDataLoaderIterator {
    batcher: Arc<dyn Batcher>,
    strategy: Box<dyn FrameBatchStrategy>,
    current_batch: usize,
    items_processed: usize,
    transforms: Arc<Pipeline>,
    total_items: usize,
    device: Device,
    
}

impl StreamingDataLoaderIterator {
    pub fn new(
        dataset: LazyFrame,
        batcher: Arc<dyn Batcher>,
        mut strategy: Box<dyn FrameBatchStrategy>,
        transforms: Arc<Pipeline>,
        total_items: usize,
        device: Device,
    ) -> Self {
        let stream = dataset
            .collect_batches(
                Engine::Streaming,
                false,
                NonZero::new(strategy.chunk_size()),
                true,
            )
            .unwrap();
        strategy.init(stream);

        Self {
            batcher,
            strategy,
            current_batch: 0,
            items_processed: 0,
            transforms,
            total_items,
            device,
        }
    }
}

impl Iterator for StreamingDataLoaderIterator {
    type Item = Result<Batch, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.current_batch += 1;

        if let Some(df) = self.strategy.batch() {
            let batch = self
                .batcher
                .batch(df, self.transforms.clone(), &self.device);
            let batch_size = batch.targets.dims()[0];
            self.items_processed += batch_size;
            return Some(Ok(batch));
        }

        None
    }
}

impl DataLoaderIterator<Batch> for StreamingDataLoaderIterator {
    fn progress(&self) -> Progress {
        Progress::new(self.items_processed, self.total_items, None)
    }
}
