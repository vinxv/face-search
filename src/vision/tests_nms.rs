#[cfg(test)]
mod tests {
    use crate::vision::retinaface::{nms, FaceBox};

    #[test]
    fn test_nms() {
        let box1 = FaceBox {
            x1: 10.0,
            y1: 10.0,
            x2: 100.0,
            y2: 100.0,
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        let box2 = FaceBox {
            x1: 15.0,
            y1: 15.0,
            x2: 95.0,
            y2: 95.0,
            score: 0.8,
            landmarks: [(0.0, 0.0); 5],
        };
        let box3 = FaceBox {
            x1: 200.0,
            y1: 200.0,
            x2: 300.0,
            y2: 300.0,
            score: 0.7,
            landmarks: [(0.0, 0.0); 5],
        };

        let boxes = vec![box1, box2, box3];
        let mut result = nms(boxes, 0.5);
        assert_eq!(result.len(), 2);

        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(result[0].score, 0.9);
        assert_eq!(result[1].score, 0.7);
    }
}
