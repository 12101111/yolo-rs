use crate::{tvm::*, yolo::Detection};
use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgb, RgbImage};
use ndarray::{Array, ArrayD, Axis};
use sdl2;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::{Color, PixelFormat, PixelFormatEnum};
use sdl2::rect::Rect;
use std::sync::mpsc;

pub fn process_input(img: DynamicImage) -> ArrayD<f32> {
    let img = img.resize(SIZE, SIZE, FilterType::Nearest);
    let (w, h) = (img.width(), img.height());
    let img = img.to_rgb();
    let img = match (w == SIZE, h == SIZE) {
        (true, true) => img,
        (false, false) => unreachable!("Resize failed"),
        (true, false) => {
            let mut newimg = RgbImage::from_pixel(SIZE, SIZE, Rgb([128, 128, 128]));
            let newy = (SIZE - h) / 2;
            newimg.copy_from(&img, 0, newy).unwrap();
            newimg
        }
        (false, true) => {
            let mut newimg = RgbImage::from_pixel(SIZE, SIZE, Rgb([128, 128, 128]));
            let newx = (SIZE - w) / 2;
            newimg.copy_from(&img, newx, 0).unwrap();
            newimg
        }
    };
    let uint_arr = Array::from(img.to_vec());
    let arr = uint_arr.mapv(|u| u as f32 / 255.0);
    let arr = arr.into_shape((USIZE, USIZE, 3)).unwrap(); // HWc 416x416x3
    let arr = arr.permuted_axes((2, 0, 1)).into_dyn(); // HWc to cHW
    arr.insert_axis(Axis(0)) // Yolov3 require 1x3x416x416
}

const FONT: &str = "/usr/share/fonts/noto/NotoSansDisplay-Regular.ttf";

pub fn display_rgb(rx: mpsc::Receiver<(Vec<u8>, (u32, u32), Vec<Detection>)>) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let ttf_context = sdl2::ttf::init().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let window = video_subsystem
        .window("Yolo Result Viewer", 480, 480)
        .position_centered()
        .build()
        .expect("Open SDL window failed");
    let mut canvas = window.into_canvas().accelerated().build().unwrap();
    let texture_creator = canvas.texture_creator();
    let font = ttf_context.load_font(FONT, 14).unwrap();
    canvas.set_draw_color(Color::RGB(127, 127, 127));
    canvas.clear();
    canvas.present();
    let names: Vec<&str> = crate::NAMES.lines().collect();
    loop {
        match event_pump.poll_event() {
            Some(Event::Quit { .. })
            | Some(Event::KeyDown {
                keycode: Some(Keycode::Escape),
                ..
            }) => break,
            Some(Event::DropFile {
                filename, ..
            }) => {
                crate::FILES.get().unwrap().send(filename).unwrap()
            }
            _ => {
                if let Ok((buf, (w, h), dets)) = rx.try_recv() {
                    canvas.window_mut().set_size(w, h).unwrap();
                    canvas.set_logical_size(w, h).unwrap();
                    let mut texture = texture_creator
                        .create_texture_streaming(PixelFormatEnum::RGB24, w, h)
                        .unwrap();
                    texture
                        .update(None, &buf, w as usize * 3)
                        .expect("Failed to update texture");
                    canvas.copy(&texture, None, None).unwrap();
                    for d in dets.iter() {
                        let b = &d.bbox;
                        let pos = b.scale_to_rect(w as i32, h as i32);
                        let mut color = rand_color();
                        canvas.set_draw_color(color);
                        canvas.draw_rect(pos.into()).unwrap();
                        color.a = 255;
                        canvas.set_draw_color(color);
                        let mut text = String::new();
                        for (c, p) in d.classes.iter() {
                            text.push_str(&format!("{}:{:.1} % ", names[*c], p.0 * 100.0));
                        }
                        let surface = font.render(&text).blended(color).unwrap();
                        let texture_font = texture_creator
                            .create_texture_from_surface(&surface)
                            .unwrap();
                        let sdl2::render::TextureQuery { width, height, .. } = texture_font.query();
                        let mut font_rect = Rect::new(pos.0, pos.1, width, height);
                        if pos.0 as u32 + width > w {
                            font_rect.set_right(w as i32);
                        }
                        if pos.1 as u32 + height > h {
                            font_rect.set_bottom(h as i32);
                        }
                        canvas.copy(&texture_font, None, font_rect).unwrap();
                    }
                    canvas.present();
                }
            }
        }
    }
    std::process::exit(0);
}

fn rand_color() -> Color {
    use core::convert::TryFrom;
    use libc::rand;
    let rand = unsafe { rand() };
    Color::from_u32(
        &PixelFormat::try_from(PixelFormatEnum::RGBA8888).unwrap(),
        rand as u32,
    )
}
