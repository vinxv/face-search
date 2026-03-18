use anyhow::Result;
use image::{DynamicImage, GenericImageView, RgbImage};
use nalgebra::Matrix3;

pub fn align_face(img: &DynamicImage, landmarks: &[(f32, f32)]) -> Result<DynamicImage> {
    // ArcFace standard reference landmarks for 112x112 image
    let reference_pts = [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ];

    let src =
        nalgebra::DMatrix::from_iterator(2, 5, landmarks.iter().flat_map(|&(x, y)| vec![x, y]));
    let dst =
        nalgebra::DMatrix::from_iterator(2, 5, reference_pts.iter().flat_map(|&[x, y]| vec![x, y]));

    // Compute simple affine transform using Procrustes-like approach / least squares.
    // For simplicity, we use nalgebra to solve `A * src = dst` where A is a 2x3 affine matrix.
    // Since src has 5 columns (landmarks), we append a row of 1s.
    let mut src_homo = nalgebra::DMatrix::<f32>::zeros(3, 5);
    for i in 0..5 {
        src_homo[(0, i)] = src[(0, i)];
        src_homo[(1, i)] = src[(1, i)];
        src_homo[(2, i)] = 1.0;
    }

    let mut dst_mat = nalgebra::DMatrix::<f32>::zeros(2, 5);
    for i in 0..5 {
        dst_mat[(0, i)] = dst[(0, i)];
        dst_mat[(1, i)] = dst[(1, i)];
    }

    // Solve for Affine matrix M: M * src_homo = dst_mat => M = dst_mat * src_homo^T * (src_homo * src_homo^T)^-1
    let src_homo_t = src_homo.transpose();
    let src_sq = &src_homo * &src_homo_t;

    let m = match src_sq.try_inverse() {
        Some(inv) => &dst_mat * &src_homo_t * inv,
        None => {
            // Fallback if matrix is singular
            return Ok(img.resize_exact(112, 112, image::imageops::FilterType::Triangle));
        }
    };

    let mut inv_m_homo = Matrix3::<f32>::identity();
    inv_m_homo[(0, 0)] = m[(0, 0)];
    inv_m_homo[(0, 1)] = m[(0, 1)];
    inv_m_homo[(0, 2)] = m[(0, 2)];
    inv_m_homo[(1, 0)] = m[(1, 0)];
    inv_m_homo[(1, 1)] = m[(1, 1)];
    inv_m_homo[(1, 2)] = m[(1, 2)];

    let inv_m = match inv_m_homo.try_inverse() {
        Some(inv) => inv,
        None => return Ok(img.resize_exact(112, 112, image::imageops::FilterType::Triangle)),
    };

    let mut aligned = RgbImage::new(112, 112);
    let (width, height) = img.dimensions();

    let rgb_img = img.to_rgb8();

    // Warp affine using backward mapping
    for y in 0..112 {
        for x in 0..112 {
            let src_x = inv_m[(0, 0)] * (x as f32) + inv_m[(0, 1)] * (y as f32) + inv_m[(0, 2)];
            let src_y = inv_m[(1, 0)] * (x as f32) + inv_m[(1, 1)] * (y as f32) + inv_m[(1, 2)];

            // Nearest neighbor interpolation for simplicity
            let src_x_idx = src_x.round() as i32;
            let src_y_idx = src_y.round() as i32;

            if src_x_idx >= 0
                && src_x_idx < width as i32
                && src_y_idx >= 0
                && src_y_idx < height as i32
            {
                let pixel = rgb_img.get_pixel(src_x_idx as u32, src_y_idx as u32);
                aligned.put_pixel(x, y, *pixel);
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(aligned))
}
