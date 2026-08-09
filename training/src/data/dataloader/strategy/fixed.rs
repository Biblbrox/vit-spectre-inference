use crate::data::{dataloader::strategy::FrameBatchStrategy, mapper::FrameMapper};
use polars::prelude::*;
use std::sync::Mutex;

pub struct FixedBatchStrategy {
    source: Option<Arc<Mutex<CollectBatches>>>,
    mapper: Option<FrameMapper>,
    shuffle: Option<u64>,
    batch_size: usize,
}

impl FixedBatchStrategy {
    pub fn new(batch_size: usize) -> Self {
        Self {
            source: None,
            mapper: None,
            shuffle: None,
            batch_size,
        }
    }

    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = Some(seed);
        self
    }

    pub fn with_mapper(mut self, mapper: FrameMapper) -> Self {
        self.mapper = Some(mapper);
        self
    }
}

impl Clone for FixedBatchStrategy {
    fn clone(&self) -> Self {
        Self {
            source: None,
            mapper: self.mapper.clone(),
            shuffle: self.shuffle,
            batch_size: self.batch_size,
        }
    }
}

impl FrameBatchStrategy for FixedBatchStrategy {
    fn init(&mut self, stream: CollectBatches) {
        self.source = Some(Arc::new(Mutex::new(stream)));
    }

    fn batch(&mut self) -> Option<DataFrame> {
        if let Some(ref stream) = self.source {
            let mut batches = stream.lock().unwrap();

            if let Some(mut batch) = batches.next().transpose().unwrap() {
                if let Some(ref mapper) = self.mapper {
                    batch = mapper(batch);
                }

                if let Some(seed) = self.shuffle {
                    batch = batch
                        .sample_n_literal(batch.height(), false, Some(true), Some(seed))
                        .unwrap();
                }

                return Some(batch);
            }
        }

        None
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn chunk_size(&self) -> usize {
        self.batch_size
    }

    fn clone_dyn(&self) -> Box<dyn FrameBatchStrategy> {
        Box::new(self.clone())
    }
}
