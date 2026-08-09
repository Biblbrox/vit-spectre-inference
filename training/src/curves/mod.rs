use burn::{
    config::Config,
    module::{AutodiffModule, Module, ModuleDisplay, ModuleDisplayDefault},
    tensor::Device,
};

#[derive(Config, Debug, Copy, PartialEq)]
pub enum SpaceCurve {
    Hilbert,
    ZOrder,
}

impl Module for SpaceCurve {
    fn visit<V: burn::module::ModuleVisitor>(&self, _visitor: &mut V) {}
    fn map<M: burn::module::ModuleMapper>(self, _mapper: &mut M) -> Self { self }
    fn to_device(self, _: &Device) -> Self { self }
    fn fork(self, _: &Device) -> Self { self }
    fn collect_devices(&self, devices: burn::module::Devices) -> burn::module::Devices { devices }
}

impl AutodiffModule for SpaceCurve {
    fn valid(&self) -> Self { *self }
    fn from_inner(module: Self) -> Self { module }
}

impl ModuleDisplayDefault for SpaceCurve {
    fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> { None }
}

impl ModuleDisplay for SpaceCurve {}

fn invert_permutation(perm: &[i32]) -> Vec<i32> {
    let mut inv = vec![0i32; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as i32;
    }
    inv
}

/// Build raster->curve and curve->raster index maps for a grid.
/// `curve_fn` maps (row, col, grid_h, grid_w) to a curve index.
fn build_permutation(
    grid_h: usize,
    grid_w: usize,
    curve_fn: impl Fn(usize, usize) -> usize,
) -> (Vec<i32>, Vec<i32>) {
    let n = grid_h * grid_w;
    let mut to_curve = vec![0i32; n];

    for row in 0..grid_h {
        for col in 0..grid_w {
            let raster = row * grid_w + col;
            // Clamp to n in case curve index exceeds grid for non-power-of-2 grids
            let curve = curve_fn(row, col) % n;
            to_curve[raster] = curve as i32;
        }
    }

    // to_curve may not be a valid permutation if curve indices collide after
    // clamping — patch duplicates by filling gaps with unassigned raster slots.
    let to_curve = repair_permutation(to_curve, n);
    let from_curve = invert_permutation(&to_curve);
    (to_curve, from_curve)
}

/// If clamping caused duplicate curve indices, assign remaining raster positions
/// to the gaps in sorted order. Keeps spatial coherence as much as possible.
fn repair_permutation(mut perm: Vec<i32>, n: usize) -> Vec<i32> {
    let mut seen = vec![false; n];
    let mut dups = Vec::new();
    for (i, &p) in perm.iter().enumerate() {
        if seen[p as usize] {
            dups.push(i);
        } else {
            seen[p as usize] = true;
        }
    }
    let mut gaps = (0..n).filter(|&i| !seen[i]);
    for dup_pos in dups {
        perm[dup_pos] = gaps.next().unwrap() as i32;
    }
    perm
}

impl SpaceCurve {
    pub fn build(&self, grid_h: usize, grid_w: usize) -> (Vec<i32>, Vec<i32>) {
        // Hilbert needs a power-of-2 order parameter
        let order = grid_h.max(grid_w).next_power_of_two();
        match self {
            SpaceCurve::Hilbert => {
                build_permutation(grid_h, grid_w, |r, c| xy_to_hilbert(r, c, order))
            }
            SpaceCurve::ZOrder => build_permutation(grid_h, grid_w, xy_to_morton),
        }
    }
}

/// Z-order (Morton) curve: interleave bits of (row, col) -> index.
/// Locality guarantee: tokens close in Z-index are within a 2×2 block.
pub fn xy_to_morton(row: usize, col: usize) -> usize {
    let mut d = 0usize;
    let bits = usize::BITS as usize;
    for i in 0..bits / 2 {
        d |= ((col >> i) & 1) << (2 * i);
        d |= ((row >> i) & 1) << (2 * i + 1);
    }
    d
}

/// Hilbert curve: stronger locality than Morton — tokens close in Hilbert
/// index are always spatially adjacent, not just within a power-of-2 block.
pub fn xy_to_hilbert(mut row: usize, mut col: usize, order: usize) -> usize {
    let mut d = 0usize;
    let mut s = order / 2;
    while s > 0 {
        let rx = usize::from(col & s > 0);
        let ry = usize::from(row & s > 0);
        d += s * s * ((3 * rx) ^ ry);
        // Rotate/reflect quadrant so curve is continuous
        if ry == 0 {
            if rx == 1 {
                col = s.wrapping_sub(1).wrapping_sub(col);
                row = s.wrapping_sub(1).wrapping_sub(row);
            }
            std::mem::swap(&mut col, &mut row);
        }
        s /= 2;
    }
    d
}
