
use burn::{
    backend::Backend,
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::Distribution,
};

#[derive(Module, Debug)]
pub struct Patcher {
    conv: Conv2d,
    
}

#[derive(Config, Debug)]
pub struct PatcherConfig {
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
}

#[derive(Module, Debug)]
pub struct PatchEmbedding {
    patcher: Patcher,
    position_embeddings: Param<Tensor<3>>,
    cls: Option<Param<Tensor<3>>>,
    dropout: Dropout,
    
}

#[derive(Config, Debug)]
pub struct PatchEmbeddingConfig {
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
    image_size: usize,
    dropout: f64,
    seq_length: usize,
    use_cls: bool,
}

impl Patcher {
    // # Shapes
    // - Images: [batch_size, num_channels, height, width]
    // - Output: [batch_size, num_channels, num_patches, embed_dim]
    pub fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        let x = self.conv.forward(images); // [batch_size, embed_dim, row_patch_num, row_patch_num]
        let x = x.flatten(2, 3); // [batch_suze, embed_dim, total_patch_num]
        x.transpose() // [batch_size, total_patch_num, embed_dim]
    }
}

impl PatcherConfig {
    pub fn init(&self, device: &Device) -> Patcher {
        Patcher {
            conv: Conv2dConfig::new(
                [self.in_channels, self.embed_dim],
                [self.patch_size, self.patch_size],
            )
            .with_stride([self.patch_size, self.patch_size])
            .init(device),
        }
    }
}

impl PatchEmbedding {
    pub fn forward(&self, images: Tensor<4>) -> Tensor<3> {
        let patches = self.patcher.forward(images); // [batch_size, total_patch_dim, embed_dim]
        let mut x = self.position_embeddings.val() + patches;
        if self.cls.is_some() {
            let [b, _, _] = x.dims();
            x = Tensor::cat(vec![self.cls.clone().unwrap().val().repeat_dim(0, b), x], 1);
        }
        self.dropout.forward(x)
    }
}

impl PatchEmbeddingConfig {
    pub fn init(&self, device: &Device) -> PatchEmbedding {
        let distribution = Distribution::Normal(0.0, 1.0);
        let cls = match self.use_cls {
            true => Some(Param::from_tensor(Tensor::<3>::random(
                [1, 1, self.embed_dim],
                distribution,
                device,
            ))),
            false => None,
        };
        PatchEmbedding {
            patcher: PatcherConfig::new(self.in_channels, self.embed_dim, self.patch_size)
                .init(device),

            position_embeddings: Param::<Tensor<3>>::from_tensor(Tensor::<3>::random(
                Shape::new([1, self.seq_length, self.embed_dim]),
                distribution,
                device,
            ))
            .set_require_grad(true),
            dropout: DropoutConfig::new(self.dropout).init(),
            cls,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Flex, flex::FlexDevice},
        Shape,
    };

    use crate::embeddings::vit::PatcherConfig;

    use super::*;

    type B = Flex;
    type Device = FlexDevice;

    const IN_CHANNELS: usize = 3;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const NUM_PATCHES: usize = (IMG_SIZE / PATCH_SIZE).pow(2); // 64
    const EMBED_DIM: usize = PATCH_SIZE.pow(2) * IN_CHANNELS;
    const BATCH_SIZE: usize = 10;
    const DROPOUT: f64 = 0.1;

    #[test]
    fn test_patcher() {
        let device = Device::default();
        let test_image = Tensor::<4>::zeros(
            Shape::new([BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );
        let patcher = PatcherConfig::new(IN_CHANNELS, EMBED_DIM, PATCH_SIZE).init(&device);
        let patched_image = patcher.forward(test_image);
        assert_eq!(
            patched_image.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES, EMBED_DIM])
        );
    }

    #[test]
    fn test_patch_embedding() {
        let device = Device::default();
        let test_image = Tensor::<4>::zeros(
            Shape::new([BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );
        let model = PatchEmbeddingConfig::new(
            IN_CHANNELS,
            EMBED_DIM,
            PATCH_SIZE,
            IMG_SIZE,
            DROPOUT,
            NUM_PATCHES,
            true,
        )
        .init(&device);
        let vit_input = model.forward(test_image);
        assert_eq!(
            vit_input.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES + 1, EMBED_DIM])
        );
    }
}
