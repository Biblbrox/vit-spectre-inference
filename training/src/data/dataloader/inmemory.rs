use std::sync::Mutex;
use std::{
    sync::{
        Arc,
        mpsc::{self, Receiver},
    },
    thread,
};

use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    backend::Backend,
};
use polars::lazy::{
    dsl::{Engine, len},
    frame::LazyFrame,
};

use crate::{
    augmentations::Pipeline,
    data::batch::{Batch, Batcher},
};

pub struct InMemoryDataLoader {
    lf: LazyFrame,
    batcher: Arc<dyn Batcher>,
    transforms: Arc<Pipeline>,
    items_total: usize,
    batch_size: usize,
    num_workers: usize,
    device: Device,
    
}

impl InMemoryDataLoader {
    pub fn new(
        lf: impl Into<LazyFrame>,
        batcher: Arc<dyn Batcher>,
        transforms: Arc<Pipeline>,
        batch_size: usize,
        num_workers: usize,
        device: Device,
    ) -> Self {
        let lf = lf.into();
        let items_total = lf
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
            lf,
            batcher,
            transforms,
            items_total,
            batch_size,
            device,
            num_workers,
        }
    }
}

impl DataLoader, Batch> for InMemoryDataLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<Batch> + 'a> {
        if self.num_workers > 0 {
            let (tx, rx) = mpsc::sync_channel(4 * self.num_workers);
            for idx in 0..self.num_workers {
                let num_workers = self.num_workers;
                let batcher = self.batcher.clone();
                let transforms = self.transforms.clone();
                let lf = self.lf.clone();
                let batch_size = self.batch_size;
                let total = self.items_total;
                let device = self.device.clone();
                let tx = tx.clone();

                thread::spawn(move || {
                    let mut offset = idx * batch_size;
                    loop {
                        let length = match offset + batch_size {
                            x if x <= total => batch_size,
                            x if x < total + batch_size => total - offset,
                            _ => 0,
                        };

                        if length > 0 {
                            let result = tx.send(
                                batcher.batch(
                                    lf.clone()
                                        .slice(offset as i64, length as u32)
                                        .collect()
                                        .unwrap(),
                                    transforms.clone(),
                                    &device,
                                ),
                            );

                            if result.is_err() {
                                break;
                            }
                        } else {
                            break;
                        }

                        offset += num_workers * batch_size;
                    }
                });
            }

            Box::new(InMemoryDataLoaderIterator {
                lf: self.lf.clone(),
                channel: Some(Arc::new(Mutex::new(rx))),
                batcher: self.batcher.clone(),
                transforms: self.transforms.clone(),
                batch_size: self.batch_size,
                items_processed: 0,
                items_total: self.items_total,
                device: &self.device,
            })
        } else {
            Box::new(InMemoryDataLoaderIterator {
                lf: self.lf.clone(),
                channel: None,
                batcher: self.batcher.clone(),
                transforms: self.transforms.clone(),
                batch_size: self.batch_size,
                items_processed: 0,
                items_total: self.items_total,
                device: &self.device,
            })
        }
    }

    fn num_items(&self) -> usize {
        self.items_total
    }

    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader, Batch>> {
        Arc::new(Self {
            lf: self.lf.clone(),
            batcher: self.batcher.clone(),
            transforms: self.transforms.clone(),
            items_total: self.items_total,
            batch_size: self.batch_size,
            num_workers: self.num_workers,
            device: device.clone(),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader, Batch>> {
        let len = end.saturating_sub(start);

        Arc::new(Self {
            lf: self.lf.clone().slice(start as i64, len as u32),
            batcher: self.batcher.clone(),
            transforms: self.transforms.clone(),
            items_total: len,
            batch_size: self.batch_size,
            num_workers: self.num_workers,
            device: self.device.clone(),
        })
    }
}

pub struct InMemoryDataLoaderIterator<'a, B: Backend> {
    lf: LazyFrame,
    channel: Option<Arc<Mutex<Receiver<Batch>>>>,
    batcher: Arc<dyn Batcher>,
    transforms: Arc<Pipeline>,
    batch_size: usize,
    items_processed: usize,
    items_total: usize,
    device: &'a B::Device,
}

impl Iterator for InMemoryDataLoaderIterator<'a, B> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(recv) = &self.channel {
            let batch = recv.lock().unwrap().recv().ok();
            let batch_size = batch.as_ref().map(|t| t.targets.dims()[0]).unwrap_or(0);
            self.items_processed += batch_size;
            batch
        } else {
            let (offset, length) = match self.items_processed + self.batch_size {
                x if x <= self.items_total => (self.items_processed, self.batch_size),
                x if x < self.items_total + self.batch_size => (
                    self.items_processed,
                    self.items_total - self.items_processed,
                ),
                _ => (0, 0),
            };

            self.items_processed += length;

            (length > 0).then_some(
                self.batcher.batch(
                    self.lf
                        .clone()
                        .slice(offset as i64, length as u32)
                        .collect()
                        .unwrap(),
                    self.transforms.clone(),
                    self.device,
                ),
            )
        }
    }
}

impl DataLoaderIterator<Batch> for InMemoryDataLoaderIterator<'a, B> {
    fn progress(&self) -> Progress {
        Progress::new(self.items_processed, self.items_total)
    }
}
