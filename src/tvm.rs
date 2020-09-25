use ::ndarray::ArrayD;
use smallvec::{smallvec, SmallVec};
use std::convert::TryInto;
use std::path::Path;
use tvm::*;

#[cfg(feature = "yolov4")]
pub const SIZE: u32 = 608;
#[cfg(not(feature = "yolov4"))]
pub const SIZE: u32 = 416;

pub const USIZE: usize = SIZE as usize;

#[cfg(feature = "yolov4")]
const SONAME: &str = "libyolov4.so";
#[cfg(not(feature = "yolov4"))]
const SONAME: &str = "libyolov3.so";

pub struct TVM {
    rt: Module,
}

impl TVM {
    pub fn init() -> TVM {
        let lib = Module::load(&Path::new(SONAME)).unwrap();
        let default = lib.get_function("default", true).unwrap();
        let ctx = Context::gpu(0);
        let ret = default.invoke(vec![ctx.into()]);
        let gmod: Module = ret.unwrap().try_into().unwrap();
        TVM { rt: gmod }
    }

    pub fn run(&self, img: ArrayD<f32>) -> Vec<NDArray> {
        let now = std::time::Instant::now();
        let ref set_input_fn = self.rt.get_function("set_input", false).unwrap();
        let input =
            NDArray::from_rust_ndarray(&img, Context::gpu(0), DataType::float(32, 1)).unwrap();
        println!(
            "input size is {:?}",
            input.shape().expect("cannot get the input shape")
        );
        set_input_fn
            .invoke(vec!["data".into(), (&input).into()])
            .unwrap();
        println!("Set input in {:?}", now.elapsed());

        let now = std::time::Instant::now();
        let ref run_fn = self.rt.get_function("run", false).unwrap();
        // execute the run function. Note that it has no argument
        run_fn.invoke(vec![]).unwrap();
        // get the `get_output` function from runtime module
        let ref get_output_fn = self.rt.get_function("get_output", false).unwrap();
        // prepare to get the output
        let mut outputs = Vec::with_capacity(12);
        for i in 0..12 {
            let dtype = if i % 2 == 0 {
                DataType::float(32, 1)
            } else {
                DataType::int(32, 1)
            };
            let output_shape: SmallVec<[usize; 4]> = match i % 4 {
                0 => {
                    let size: usize = if cfg!(feature = "yolov3") {
                        (SIZE >> 3) >> (i / 4)
                    } else {
                        (SIZE >> 5) << (i / 4)
                    } as usize;
                    smallvec![1, 255, size, size]
                }
                1 => smallvec![3],
                2 => smallvec![18],
                3 => smallvec![6],
                _ => unreachable!(),
            };
            // We alaways need to place output on CPU (main RAM)
            let output = NDArray::empty(&output_shape, Context::cpu(0), dtype);
            get_output_fn
                .invoke(vec![i.into(), (&output).into()])
                .unwrap();
            outputs.push(output);
        }
        println!("TVM run in {:?}", now.elapsed());
        outputs
    }
}
