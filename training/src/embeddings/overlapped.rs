use burn::{Tensor, config::Config, module::Module, backend::Backend, tensor::Device};

use crate::conv::{ConvBNAct, ConvBNActConfig};

#[derive(Module, Debug)]
pub struct PatchEmbeddingOverlapped {
    stem: ConvBNAct,
    proj: ConvBNAct,
    
}

impl PatchEmbeddingOverlapped {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.stem.forward(x);
        self.proj.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct PatchEmbeddingOverlappedConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 3)]
    pub stem_kernel: usize,
    #[config(default = 2)]
    pub stem_stride: usize,
    #[config(default = 3)]
    pub proj_kernel: usize,
    #[config(default = 2)]
    pub proj_stride: usize,
}

impl PatchEmbeddingOverlappedConfig {
    pub fn init(&self, device: &Device) -> PatchEmbeddingOverlapped {
        PatchEmbeddingOverlapped {
            stem: ConvBNActConfig::new(self.in_channels, self.out_channels, self.stem_kernel)
                .with_stride(self.stem_stride)
                .init(device),
            proj: ConvBNActConfig::new(self.out_channels, self.out_channels, self.proj_kernel)
                .with_stride(self.proj_stride)
                .init(device),
        }
    }
}

#[cfg(test)]
mod tests {

    use burn::{backend::Flex, prelude::*};

    use crate::embeddings::overlapped::PatchEmbeddingOverlappedConfig;

    type B = Flex;

    #[test]
    fn patch_embedding_reduces_spatial_size() {
        let device = Default::default();

        let cfg = PatchEmbeddingOverlappedConfig::new(3, 64)
            .with_stem_kernel(3)
            .with_stem_stride(2)
            .with_proj_kernel(3)
            .with_proj_stride(2);

        let model = cfg.init(&device);

        let x = Tensor::<4>::zeros([2, 3, 224, 224], &device);
        let y = model.forward(x);

        assert_eq!(y.dims(), [2, 64, 56, 56]);
    }
}
