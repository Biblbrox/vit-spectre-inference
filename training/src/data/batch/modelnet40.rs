use std::sync::Arc;

use burn::{
    prelude::{Device, Tensor},
    tensor::{DType, Int, TensorData},
};
use polars::prelude::*;

use crate::data::batch::{Batch, Batcher};

const POINTCOL: &str = "image";
const LABELCOL: &str = "label";

const NUM_POINTS: usize = 1024;
const NUM_CHANNELS: usize = 3;

pub struct ModelNet40Batcher;

impl ModelNet40Batcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl Batcher for ModelNet40Batcher {
    fn batch(
        &self,
        df: DataFrame,
        _transforms: Arc<crate::augmentations::Pipeline>,
        device: &Device,
    ) -> Batch {
        let batch_size = df.height();

        let total_floats = batch_size * NUM_POINTS * NUM_CHANNELS;
        let mut pointbuf: Vec<u8> = Vec::with_capacity(total_floats * 4);

        df.column(POINTCOL)
            .unwrap()
            .binary()
            .unwrap()
            .iter()
            .flatten()
            .for_each(|chunk| pointbuf.extend_from_slice(chunk));

        let pointdata = TensorData::from_bytes_vec(
            pointbuf,
            [batch_size, NUM_POINTS, NUM_CHANNELS],
            DType::F32,
        );

        let points = Tensor::<3>::from_data(pointdata, device);

        let labelbuf: Vec<i32> = df
            .column(LABELCOL)
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let targets = Tensor::<1, Int>::from_ints(labelbuf.as_slice(), device);

        Batch {
            data: points.flatten::<1>(0, -1),
            targets,
        }
    }
}
