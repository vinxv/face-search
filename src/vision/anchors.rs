#[derive(Debug, Clone)]
pub struct Anchor {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

pub fn generate_anchors(height: usize, width: usize) -> Vec<Anchor> {
    let min_sizes = [[16.0, 32.0], [64.0, 128.0], [256.0, 512.0]];
    let steps = [8.0, 16.0, 32.0];

    let mut anchors = Vec::new();

    for (k, step) in steps.iter().enumerate() {
        let feature_map_height = (height as f32 / step).ceil() as usize;
        let feature_map_width = (width as f32 / step).ceil() as usize;

        for i in 0..feature_map_height {
            for j in 0..feature_map_width {
                let cy = (i as f32 + 0.5) * step;
                let cx = (j as f32 + 0.5) * step;

                for min_size in &min_sizes[k] {
                    let w = min_size;
                    let h = min_size;

                    anchors.push(Anchor {
                        x: cx,
                        y: cy,
                        w: *w,
                        h: *h,
                    });
                }
            }
        }
    }
    anchors
}
