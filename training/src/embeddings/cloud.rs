
//! [TODO:description]

use burn::{
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{Device, Distribution, Int, activation::relu},
};

#[derive(Module, Debug)]
pub struct CloudPatcher {
    mlp1: Linear,
    mlp2: Linear,
    num_centers: usize,
    k_neighbours: usize,
    density_radius: f32,
    
}

#[derive(Config, Debug)]
pub struct CloudPatcherConfig {
    pub num_centers: usize,
    pub k_neighbours: usize,
    pub density_radius: f32,
    pub embed_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
}

impl CloudPatcher {
    // # Shapes
    // - Points: [batch_size, num_points, 3]
    // - Output: [batch_size, num_centers, embed_dim]
    pub fn forward(&self, points: Tensor<3>) -> Tensor<3> {
        let [b, _n, _] = points.dims();
        let m = self.num_centers;
        let k = self.k_neighbours;

        // 1. Density per point [B, N]
        let density = estimate_density(points.clone(), self.density_radius);

        // 2. Patch centres via density-weighted FPS → [B, M]
        let center_idx = farthest_point_sample(points.clone(), density.clone(), m);

        // 3. Gather centre coordinates [B, M, 3]
        let center_idx_exp = center_idx.clone().reshape([b, m, 1]).expand([b, m, 3]);
        let centers = points.clone().gather(1, center_idx_exp);

        // 4. KNN grouping in local frame [B, M, K, 3]
        let grouped = knn_group(points, centers, k);

        // 5. Density at each centre [B, M]
        let center_density = density.gather(1, center_idx.reshape([b, m]));

        // 6. Density-aware local normalisation
        let grouped = density_normalise(grouped, center_density);

        // 7. Mini-PointNet: shared MLP over [B*M, K, 3] + max pool over K
        let flat = grouped.reshape([b * m, k, 3]);

        let feat = relu(self.mlp1.forward(flat)); // [B*M, K, hidden_dim]
        let feat = relu(self.mlp2.forward(feat)); // [B*M, K, embed_dim]

        feat.max_dim(1) // [B*M, 1, embed_dim]
            .squeeze_dim::<2>(1) // [B*M, embed_dim]
            .reshape([b, m, self.mlp2.weight.dims()[1]])
    }
}

impl CloudPatcherConfig {
    pub fn init(&self, device: &Device) -> CloudPatcher {
        CloudPatcher {
            mlp1: LinearConfig::new(3, self.hidden_dim).init(device),
            mlp2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
            num_centers: self.num_centers,
            k_neighbours: self.k_neighbours,
            density_radius: self.density_radius,
        }
    }
}

/**
* Embedding layer, which transforms point cloud data
* into conventional patch embeddings, which ViT can process
*/
#[derive(Module, Debug)]
pub struct CloudPatchEmbedding {
    patcher: CloudPatcher,
    position_embeddings: Param<Tensor<3>>, // [1, M, embed_dim]
    cls: Option<Param<Tensor<3>>>,
    dropout: Dropout,
    
}

#[derive(Config, Debug)]
pub struct CloudPatchEmbeddingConfig {
    pub num_centers: usize,
    pub k_neighbours: usize,
    pub density_radius: f32,
    pub embed_dim: usize,
    pub dropout: f64,
    pub use_cls: bool,
    #[config(default = 64)]
    pub hidden_dim: usize,
}

impl CloudPatchEmbedding {
    // # Shapes
    // - Points: [batch_size, num_points, 3]
    // - Output: [batch_size, num_centers (+ 1 if cls), embed_dim]
    pub fn forward(&self, points: Tensor<3>) -> Tensor<3> {
        let patches = self.patcher.forward(points); // [B, M, embed_dim]
        let mut x = self.position_embeddings.val() + patches;

        if let Some(ref cls) = self.cls {
            let [b, _, _] = x.dims();
            x = Tensor::cat(vec![cls.val().repeat_dim(0, b), x], 1);
        }

        self.dropout.forward(x)
    }
}

impl CloudPatchEmbeddingConfig {
    pub fn init(&self, device: &Device) -> CloudPatchEmbedding {
        let distribution = Distribution::Normal(0.0, 1.0);

        let cls = if self.use_cls {
            Some(Param::from_tensor(Tensor::<3>::random(
                [1, 1, self.embed_dim],
                distribution,
                device,
            )))
        } else {
            None
        };

        CloudPatchEmbedding {
            patcher: CloudPatcherConfig::new(
                self.num_centers,
                self.k_neighbours,
                self.density_radius,
                self.embed_dim,
            )
            .with_hidden_dim(self.hidden_dim)
            .init(device),
            position_embeddings: Param::from_tensor(Tensor::<3>::random(
                [1, self.num_centers, self.embed_dim],
                distribution,
                device,
            ))
            .set_require_grad(true),
            dropout: DropoutConfig::new(self.dropout).init(),
            cls,
        }
    }
}

fn estimate_density(points: Tensor<3>, radius: f32) -> Tensor<2> {
    let [b, n, _] = points.dims();
    let p1 = points.clone().unsqueeze_dim::<4>(2).expand([b, n, n, 3]);
    let p2 = points.clone().unsqueeze_dim::<4>(1).expand([b, n, n, 3]);
    let dist2 = (p1 - p2).powf_scalar(2.0).sum_dim(3).squeeze_dim::<3>(3);
    dist2
        .lower_equal_elem(radius * radius)
        .float()
        .sum_dim(2)
        .squeeze_dim(2)
}

fn farthest_point_sample(
    points: Tensor<3>,
    density: Tensor<2>,
    num_centers: usize,
) -> Tensor<2, Int> {
    let [b, n, _] = points.dims();
    let device = points.device();

    let mut min_dist = Tensor::<2>::full([b, n], f32::MAX, &device);
    let mut selected: Vec<Tensor<2, Int>> = Vec::with_capacity(num_centers);
    let mut current = Tensor::<1, Int>::zeros([b], &device);

    for _ in 0..num_centers {
        selected.push(current.clone().unsqueeze_dim(1)); // [B, 1]

        let cur_idx = current.clone().reshape([b, 1, 1]).expand([b, 1, 3]);
        let cur_pts = points.clone().gather(1, cur_idx); // [B, 1, 3]
        let diff = points.clone() - cur_pts.expand([b, n, 3]);
        let dist2 = diff.powf_scalar(2.0).sum_dim(2).squeeze_dim(2);

        min_dist = min_dist.clone().min_pair(dist2);
        current = (min_dist.clone() / (density.clone() + 1e-6))
            .argmax(1)
            .reshape([b]);
    }

    Tensor::cat(selected, 1) // [B, M]
}

fn knn_group(points: Tensor<3>, centers: Tensor<3>, k: usize) -> Tensor<4> {
    let [b, n, _] = points.dims();
    let [_, m, _] = centers.dims();

    let c = centers.clone().unsqueeze_dim::<4>(2).expand([b, m, n, 3]);
    let p = points.clone().unsqueeze_dim::<4>(1).expand([b, m, n, 3]);
    let dist2 = (c - p).powf_scalar(2.0).sum_dim(3).squeeze_dim::<3>(3); // [B, M, N]

    let knn_idx = dist2.argtopk(k, 2); // [B, M, K]

    let idx_exp = knn_idx.reshape([b, m * k, 1]).expand([b, m * k, 3]);
    let grouped = points.gather(1, idx_exp).reshape([b, m, k, 3]);

    let centers_exp = centers.unsqueeze_dim::<4>(2).expand([b, m, k, 3]);
    grouped - centers_exp
}

fn density_normalise(
    grouped: Tensor<4>,
    center_density: Tensor<2>,
) -> Tensor<4> {
    let [b, m, k, _] = grouped.dims();
    let scale = (center_density + 1e-6)
        .sqrt()
        .reshape([b, m, 1, 1])
        .expand([b, m, k, 3]);
    grouped / scale
}
