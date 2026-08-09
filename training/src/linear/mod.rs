use burn::{Tensor, module::Module, nn::Linear, backend::Backend};

use crate::{linear::monarch::MonarchLinear, utils::gcd};

pub mod monarch;

pub fn optimal_block_count(in_features: usize, out_features: usize) -> usize {
    let g = gcd(in_features, out_features);
    let target = (in_features.min(out_features) as f64).sqrt() as usize;

    // All divisors of gcd, sorted by distance from sqrt — prefer multiples of 16
    let mut divisors: Vec<usize> = (1..=g).filter(|&d| g.is_multiple_of(d)).collect();
    divisors.sort_by_key(|&d| {
        let b = in_features / d;
        let e = out_features / d;
        let distance = (d as isize - target as isize).unsigned_abs();

        let alignment_penalty = match (b % 16, e % 16) {
            (0, 0) => 0,          // both 16-aligned — ideal for tensor cores
            (0, _) | (_, 0) => 1, // one side aligned
            (b_rem, e_rem) if b_rem % 8 == 0 || e_rem % 8 == 0 => 2, // 8-aligned fallback
            _ => 3,
        };

        (alignment_penalty, distance)
    });

    divisors[0]
}

#[derive(Module, Debug)]
pub enum LinearLayer {
    Dense(Linear),
    Monarch(MonarchLinear),
}

impl LinearLayer {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Monarch(l) => l.forward(x),
        }
    }
}
