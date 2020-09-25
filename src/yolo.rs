use crate::tvm::*;
use float_ord::FloatOrd;
use ndarray::{Array, Dim};
use std::collections::BTreeMap;
use std::fmt;
use tvm::NDArray;

const THRESH: f32 = 0.5;
pub struct Out {
    classes: usize,
    data: Array<f32, Dim<[usize; 4]>>,
    mask: Vec<i32>,
    biases: Vec<f32>,
    imw: f64,
    imh: f64,
}

#[derive(Debug, Clone)]
pub struct BBox {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl BBox {
    pub fn left(&self) -> f64 {
        self.x - self.w / 2.0
    }
    pub fn right(&self) -> f64 {
        self.x + self.w / 2.0
    }
    pub fn top(&self) -> f64 {
        self.y - self.h / 2.0
    }
    pub fn bot(&self) -> f64 {
        self.y + self.h / 2.0
    }
    fn overlay(&self, rhs: &BBox) -> f64 {
        let left = self.left().max(rhs.left());
        let right = self.right().min(rhs.right());
        let w = (right - left).max(0.0);
        let top = self.top().max(rhs.top());
        let bot = self.bot().min(rhs.bot());
        let h = (bot - top).max(0.0);
        w * h
    }
    fn union(&self, rhs: &BBox) -> f64 {
        self.w * self.h + rhs.w * rhs.h - self.overlay(rhs)
    }
    fn iou(&self, rhs: &BBox) -> f64 {
        self.overlay(rhs) / self.union(rhs)
    }
    pub fn scale_to_rect(&self, imw: i32, imh: i32) -> (i32, i32, u32, u32) {
        let w = imw as f64;
        let h = imh as f64;
        let left = ((self.left() * w) as i32).max(0);
        let right = ((self.right() * w) as i32).min(imw - 1);
        let top = ((self.top() * h) as i32).max(0);
        let bot = ((self.bot() * h) as i32).min(imh - 1);
        (left, top, (right - left) as u32, (bot - top) as u32)
    }
}

#[test]
fn iou() {
    let b1 = BBox {
        x: 0.5,
        y: 0.5,
        w: 1.0,
        h: 1.0,
    };
    assert_eq!(b1.left(), 0.0);
    assert_eq!(b1.right(), 1.0);
    assert_eq!(b1.top(), 0.0);
    assert_eq!(b1.bot(), 1.0);
    assert_eq!(b1.overlay(&b1), 1.0);
    assert_eq!(b1.union(&b1), 1.0);
    assert_eq!(b1.iou(&b1), 1.0);
}

#[derive(Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub classes: BTreeMap<usize, FloatOrd<f32>>,
    pub objectness: f32,
}

pub struct DetectionScale<'a>(pub &'a Detection,pub i32,pub i32);

impl std::fmt::Display for DetectionScale<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = self.0.bbox.scale_to_rect(self.1, self.2);
        write!(f, "x:{} y:{} w:{} h:{}", pos.0, pos.1, pos.2, pos.3)?;
        for (c, p) in self.0.classes.iter() {
            write!(f, " [{} {}],", c, p.0)?;
        }
        writeln!(f, "")
    }
}

impl Out {
    pub fn new(arrs: &[NDArray], imw: u32, imh: u32) -> Out {
        assert_eq!(arrs.len(), 4);
        let arr = arrs[0].to_vec::<f32>().unwrap(); // 1x255x52x52
        let mask = arrs[1].to_vec::<i32>().unwrap(); // 3
        let biases = arrs[2].to_vec::<f32>().unwrap(); // 18
                                                       // n, out_c, out_h, out_w, classes, total
        let attrs = arrs[3].to_vec::<i32>().unwrap(); // 6
                                                      // 1 x 255 x 52 x52 => 3 x 85 x 52 x 52
                                                      // 85: xywh anchor box offset, objectness score, classes
        let shape = [
            attrs[0] as usize,
            (attrs[1] / attrs[0]) as usize,
            attrs[2] as usize,
            attrs[3] as usize,
        ];
        let classes = attrs[4] as usize;
        let data = Array::from_shape_vec(shape, arr).unwrap();
        Out {
            classes,
            data,
            mask,
            biases,
            imw: imw as f64,
            imh: imh as f64,
        }
    }
    fn get_box(&self, c: usize, h: usize, w: usize, n: usize) -> BBox {
        let shape = self.data.shape();
        let bx = (w as f64 + self.data[[c, 0, h, w]] as f64) / shape[2] as f64;
        let by = (h as f64 + self.data[[c, 1, h, w]] as f64) / shape[3] as f64;
        let bw = self.data[[c, 2, h, w]].exp() as f64 * self.biases[2 * n] as f64 / SIZE as f64;
        let bh = self.data[[c, 3, h, w]].exp() as f64 * self.biases[2 * n + 1] as f64 / SIZE as f64;
        BBox {
            x: bx,
            y: by,
            w: bw,
            h: bh,
        }
    }
    pub fn get_yolo_detections(&self) -> Vec<Detection> {
        let shape = self.data.shape();
        let mut dets = Vec::new();
        for c in 0..shape[0] {
            let n = self.mask[c];
            assert!(n >= 0 && n < 9);
            for h in 0..shape[2] {
                for w in 0..shape[3] {
                    let objectness = self.data[[c, 4, h, w]];
                    if objectness > THRESH {
                        let bbox = self.get_box(c, h, w, n as usize);
                        let mut classes = BTreeMap::new();
                        for class in 0..self.classes {
                            let prob = objectness * self.data[[c, 5 + class, h, w]];
                            if prob >= THRESH {
                                classes.insert(class, FloatOrd(prob));
                            }
                        }
                        if classes.len() > 0 {
                            dets.push(Detection {
                                bbox,
                                classes,
                                objectness,
                            })
                        }
                    }
                }
            }
        }
        dets.iter_mut().for_each(|d| {
            let mut bbox = &mut d.bbox;
            // Only with WEIGHT==HEIGHT
            if self.imw < self.imh {
                bbox.x = 0.5 + (bbox.x - 0.5) * self.imh / self.imw;
                bbox.w = bbox.w * self.imh / self.imw;
            } else if self.imw > self.imh {
                bbox.y = 0.5 + (bbox.y - 0.5) * self.imw / self.imh;
                bbox.h = bbox.h * self.imw / self.imh;
            };
        });
        dets
    }
}

const NMS_THRESH: f64 = 0.45;

pub fn nms_sort(mut dets: Vec<Detection>) -> Vec<Detection> {
    let mut ans = Vec::new();
    while !dets.is_empty() {
        dets.sort_by_key(|d| *d.classes.values().max().unwrap_or(&FloatOrd(0.0)));
        ans.push(dets.pop().unwrap());
        dets = dets
            .into_iter()
            .filter(|d| d.bbox.iou(&ans.last().unwrap().bbox) < NMS_THRESH)
            .collect();
    }
    ans
}
