use burn::{Tensor, backend::Backend};

// For now, now backwards... I'm working on it
//pub mod backward;
// CubeCL API is very unstble.. Uff..
//pub mod launch;
//pub mod monarch;

pub fn monarch_fused_reference(
    x: Tensor<3>,
    left: Tensor<3>,
    right: Tensor<3>,
    a: usize,
    b: usize,
    bias: Option<Tensor<3>>,
) -> Tensor<3> {
    let [batch, n, _] = x.dims();

    // [B,N,a*b] -> [BN,a,b] -> [a,b,BN]
    let x = x.reshape([-1, a as i32, b as i32]).permute([1, 2, 0]); // [a, b, BN]

    // Left: [a,c,b] @ [a,b,BN] -> [a,c,BN]
    let x = left.matmul(x); // [a, c, BN]

    // Transpose block structure: [a,c,BN] -> [c,a,BN]
    let x = x.swap_dims(0, 1); // [c, a, BN]

    // Right: [c,d,a] @ [c,a,BN] -> [c,d,BN]
    let x = right.matmul(x); // [c, d, BN]

    // [c,d,BN] -> [BN,c,d] -> [B,N,out_features]
    let x = x
        .permute([2, 0, 1]) // [BN, c, d]
        .reshape([batch as i32, n as i32, -1]);

    match bias {
        Some(bias) => x + bias,
        None => x,
    }
}
