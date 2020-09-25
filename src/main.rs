use image::io::Reader;
use image::GenericImageView;
use once_cell::sync::OnceCell;
use std::sync::mpsc;
mod images;
mod tvm;
mod yolo;
use yolo::*;

const NAMES: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/coco.names"));

pub static FILES: OnceCell<std::sync::mpsc::SyncSender<String>> = OnceCell::new();

pub static RESULT: OnceCell<std::sync::mpsc::SyncSender<(Vec<u8>, (u32, u32), Vec<Detection>)>> =
    OnceCell::new();

pub fn main() {
    let (ftx, frx) = mpsc::sync_channel(1);
    FILES.set(ftx).unwrap();
    let (rtx, rrx) = std::sync::mpsc::sync_channel(1);
    RESULT.set(rtx).unwrap();
    let gui = std::thread::spawn(move || crate::images::display_rgb(rrx));
    let now = std::time::Instant::now();
    let rt = crate::tvm::TVM::init();
    println!("TVM initialized in {:?}", now.elapsed());
    println!("Drag file to Window");
    while let Ok(file) = frx.recv() {
        let img = if let Ok(img) = Reader::open(&file) {
            if let Ok(img) = img.decode() {
                img
            } else {
                eprintln!("Image can't decode");
                continue;
            }
        } else {
            eprintln!("Open image failed");
            continue;
        };
        let (imw, imh) = (img.width(), img.height());
        print!("Image size: {}x{}\n", imw, imh);
        let now = std::time::Instant::now();
        let imgbuf = img.to_rgb();
        let img = crate::images::process_input(img);
        println!("Image processed in {:?}", now.elapsed());
        let outputs = rt.run(img);
        let now = std::time::Instant::now();
        let mut dets = Vec::new();
        for i in 0..3 {
            let out = crate::yolo::Out::new(&outputs[i * 4..i * 4 + 4], imw, imh);
            dets.extend(out.get_yolo_detections());
        }
        println!(
            "Get detection result in {:?}, {} result before nms sort",
            now.elapsed(),
            dets.len()
        );
        let now = std::time::Instant::now();
        let dets = crate::yolo::nms_sort(dets);
        println!("NMS sort in {:?}", now.elapsed());
        let mut res = String::new();
        for d in dets.iter() {
            res.push_str(&DetectionScale(&d, imw as i32, imh as i32).to_string());
        }
        print!("{}", res);
        RESULT
            .get()
            .unwrap()
            .send((imgbuf.to_vec(), (imw, imh), dets))
            .unwrap();
    }
    gui.join().unwrap();
}
