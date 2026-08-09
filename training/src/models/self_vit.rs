use burn::{
    backend::Backend,
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::activation::{gelu, softmax},
};

use crate::models::fast_vit::{FastViT, FastViTConfig};

#[derive(Module, Debug)]
pub struct ProjectionHead {
    fc1: Linear,
    fc2: Linear,
    
}

#[derive(Config, Debug)]
pub struct ProjectionHeadConfig {
    embed_dim: usize,
    proto_dim: usize,
}

impl ProjectionHeadConfig {
    pub fn init(&self, device: &Device) -> ProjectionHead {
        ProjectionHead {
            fc1: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            fc2: LinearConfig::new(self.embed_dim, self.proto_dim).init(device),
        }
    }
}

impl ProjectionHead {
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = gelu(self.fc1.forward(x));
        let x = self.fc2.forward(x);
        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp(1e-8, f64::MAX);
        x / norm
    }
}

#[derive(Module, Debug)]
pub struct SelfViTBranch {
    pub backbone: FastViT,
    pub head: ProjectionHead,
    
}

impl SelfViTBranch {
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        self.head.forward(self.backbone.forward(images))
    }
}

#[derive(Module, Debug)]
pub struct SelfViT {
    pub student: SelfViTBranch,
    pub teacher: SelfViTBranch,
    center: Param<Tensor<2>>,
    pub temp_student: f32,
    pub temp_teacher: f32,
    pub center_momentum: f32,
    
}

#[derive(Config, Debug)]
pub struct SelfViTConfig {
    in_channels: usize,
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    num_classes: usize,
    patch_size: usize,
    image_size: usize,
    hid_dim: usize,
    dropout: f64,
    sinkhorn_temp: f32,
    proto_dim: usize,
    #[config(default = 0.1)]
    temp_student: f32,
    #[config(default = 0.04)]
    temp_teacher: f32,
    #[config(default = 0.9)]
    center_momentum: f32,
}

impl SelfViTConfig {
    pub fn init(&self, device: &Device) -> SelfViT {
        let backbone_cfg = FastViTConfig::new(
            self.in_channels,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.num_classes,
            self.patch_size,
            self.image_size,
            self.hid_dim,
            self.dropout,
            self.sinkhorn_temp,
        );
        let head_cfg = ProjectionHeadConfig::new(self.num_classes, self.proto_dim);

        let make_branch = || SelfViTBranch {
            backbone: backbone_cfg.init(device),
            head: head_cfg.init(device),
        };

        SelfViT {
            student: make_branch(),
            teacher: make_branch(),
            center: Param::from_tensor(Tensor::<2>::zeros([1, self.proto_dim], device))
                .set_require_grad(false),
            temp_student: self.temp_student,
            temp_teacher: self.temp_teacher,
            center_momentum: self.center_momentum,
        }
    }
}

impl SelfViT {
    pub fn forward(&self, images: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
        let student_out = self.student.forward(images.clone());
        let teacher_out = self.student.forward(images);

        (student_out, teacher_out)
    }

    pub fn update_center(&mut self, teacher_global_mean: Tensor<2>) {
        let m = self.center_momentum as f64;
        let new_center = self.center.val() * m + teacher_global_mean * (1.0 - m);
        self.center = Param::from_tensor(new_center).set_require_grad(false);
    }
}
