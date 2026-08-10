
use burn::{
    prelude::Tensor,
    tensor::Int,
};

use crate::utils::sample_beta;

pub struct CutMixOutput {
    pub images: Tensor<4>,
    pub labels_a: Tensor<1, Int>,
    pub labels_b: Tensor<1, Int>,
    pub lambda: f32,
    
}

/// CutMix (Yun et al. 2019).
pub struct CutMix {
    alpha: f64,
}

impl CutMix {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn apply(
        &self,
        images: Tensor<4>,
        labels: Tensor<1, Int>,
    ) -> CutMixOutput {
        let mut rng = fastrand::Rng::new();
        let [b, c, h, w] = [
            images.dims()[0],
            images.dims()[1],
            images.dims()[2],
            images.dims()[3],
        ];
        let device = images.device();

        let lambda = sample_beta(&mut rng, self.alpha, self.alpha) as f32;

        let cut_h = ((h as f32) * (1.0 - lambda).sqrt()) as usize;
        let cut_w = ((w as f32) * (1.0 - lambda).sqrt()) as usize;

        let cx = rng.usize(0..w);
        let cy = rng.usize(0..h);

        let x1 = cx.saturating_sub(cut_w / 2).min(w - cut_w.max(1));
        let y1 = cy.saturating_sub(cut_h / 2).min(h - cut_h.max(1));
        let x2 = (x1 + cut_w).min(w);
        let y2 = (y1 + cut_h).min(h);

        let actual_lambda = 1.0 - ((y2 - y1) * (x2 - x1)) as f32 / (h * w) as f32;

        // Shuffle indices for the second image in each pair
        let shuffled_idx: Vec<i32> = {
            let mut idx: Vec<i32> = (0..b as i32).collect();
            // Fisher-Yates
            for i in (1..b).rev() {
                let j = rng.usize(0..=i);
                idx.swap(i, j);
            }
            idx
        };
        let perm = Tensor::<1, Int>::from_ints(shuffled_idx.as_slice(), &device);

        // Shuffled batch
        let images_b = images.clone().select(0, perm.clone());
        let labels_b = labels.clone().select(0, perm);

        // Build mask: 1 everywhere, 0 inside the cut box
        let mut mask_data = vec![1f32; b * c * h * w];
        for bi in 0..b {
            for ci in 0..c {
                for hi in y1..y2 {
                    for wi in x1..x2 {
                        let idx = bi * (c * h * w) + ci * (h * w) + hi * w + wi;
                        mask_data[idx] = 0.0;
                    }
                }
            }
        }
        let mask = Tensor::<1>::from_floats(mask_data.as_slice(), &device).reshape([b, c, h, w]);

        // mixed = images_a * mask + images_b * (1 - mask)
        let mixed = images.clone() * mask.clone() + images_b * (mask.neg() + 1.0);

        CutMixOutput {
            images: mixed,
            labels_a: labels,
            labels_b,
            lambda: actual_lambda,
        }
    }
}

pub struct MixUpOutput {
    pub images: Tensor<4>,
    pub labels_a: Tensor<1, Int>,
    pub labels_b: Tensor<1, Int>,
    pub lambda: f32,
    
}

pub struct MixUp {
    alpha: f64,
}

impl MixUp {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn apply(
        &self,
        images: Tensor<4>,
        labels: Tensor<1, Int>,
    ) -> MixUpOutput {
        let mut rng = fastrand::Rng::new();
        let b = images.dims()[0];
        let device = images.device();

        let lambda = sample_beta(&mut rng, self.alpha, self.alpha) as f32;

        let shuffled_idx: Vec<i32> = {
            let mut idx: Vec<i32> = (0..b as i32).collect();
            for i in (1..b).rev() {
                let j = rng.usize(0..=i);
                idx.swap(i, j);
            }
            idx
        };
        let perm = Tensor::<1, Int>::from_ints(shuffled_idx.as_slice(), &device);

        let images_b = images.clone().select(0, perm.clone());
        let labels_b = labels.clone().select(0, perm);

        let mixed = images.clone() * lambda + images_b * (1.0 - lambda);

        MixUpOutput {
            images: mixed,
            labels_a: labels,
            labels_b,
            lambda,
        }
    }
}
