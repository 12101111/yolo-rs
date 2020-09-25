# Run Yolo using TVM and rust frontend on NVIDIA jetson agx xavier or PC

## Build TVM

[Following the offical docs to build TVM](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github)

Please get source code using git. We will merge some patch using git later.

## Build darknet (Yolo v3)

TVM currently only support yolo v3. We need to checkout a history commit, master branch don't work.

```shell
git clone https://github.com/pjreddie/darknet.git
cd darknet
git checkout f6d861736038da22c9eb0739dca84003c5a5e275
make -j8
```

We would get a `libdarknet.so` after built it.

Download the weight file.

```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

## Compile yolov3 using TVM

```python
#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing.darknet import __darknetffi__

cfg_path = './cfg/yolov3.cfg'
weights_path = './yolov3.weights'
lib_path = './libdarknet.so'

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
target = 'llvm'
target_host = 'llvm'

print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

lib.export_library('deploy_lib.so')

with open("deploy_graph.json", "w") as fo:
    fo.write(graph)

with open("deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))
```

## Rust frontend

```rust
const WIDTH: u32 = 416;
const HEIGHT: u32 = 416;

fn run(img: ArrayD<f32>) -> Vec<NDArray> {
    // get the global TVM graph runtime function
    let runtime_create_fn = Function::get("tvm.graph_runtime.create").unwrap();
    let lib = Module::load(&Path::new("deploy_lib.so")).unwrap();
    let graph = fs::read_to_string("deploy_graph.json").unwrap();
    let runtime_create_fn_ret = runtime_create_fn.invoke(vec![
        graph.into(),
        (&lib).into(),
        (DeviceType::CPU as i32).into(),
        0i32.into(),
    ]);
    // get graph runtime module
    let gmod: Module = runtime_create_fn_ret.unwrap().try_into().unwrap();
    let ref load_param_fn = gmod.get_function("load_params", false).unwrap();
    // parse parameters and convert to TVMByteArray
    let params: Vec<u8> = fs::read("deploy_param.params").unwrap();
    let barr = ByteArray::from(&params);
    load_param_fn.invoke(vec![(&barr).into()]).unwrap();
    let now = std::time::Instant::now();
    // get the set_input function
    let ref set_input_fn = gmod.get_function("set_input", false).unwrap();

    let input = NDArray::from_rust_ndarray(
        &img,
        Context::new(DeviceType::CPU, 0),
        DataType::float(32, 1),
    )
    .unwrap();
    println!(
        "input size is {:?}",
        input.shape().expect("cannot get the input shape")
    );
    set_input_fn
        .invoke(vec!["data".into(), (&input).into()])
        .unwrap();

    // get `run` function from runtime module
    let ref run_fn = gmod.get_function("run", false).unwrap();
    // execute the run function. Note that it has no argument
    run_fn.invoke(vec![]).unwrap();
    // get the `get_output` function from runtime module
    let ref get_output_fn = gmod.get_function("get_output", false).unwrap();

    // prepare to get the output
    let mut outputs = Vec::with_capacity(12);
    for i in 0..12 {
        let dtype = if i % 2 == 0 {
            DataType::float(32, 1)
        } else {
            DataType::int(32, 1)
        };
        let output_shape: SmallVec<[usize; 4]> = match i % 4 {
            0 => smallvec![
                1,
                255,
                HEIGHT as usize >> (3 + (i / 4)),
                WIDTH as usize >> (3 + (i / 4))
            ],
            1 => smallvec![3],
            2 => smallvec![18],
            3 => smallvec![6],
            _ => unreachable!(),
        };
        let output = NDArray::empty(&output_shape, Context::new(DeviceType::CPU, 0), dtype);
        get_output_fn
            .invoke(vec![i.into(), (&output).into()])
            .unwrap();
        outputs.push(output);
    }
    println!("TVM run in {:?}", now.elapsed());
    outputs
}
```

Build in release mode and check the binary.

```
$ ldd target/release/yolov3-rs
	linux-vdso.so.1 (0x0000007fadaf5000)
	libtvm.so => not found
	libdl.so.2 => /lib/aarch64-linux-gnu/libdl.so.2 (0x0000007fad94a000)
	libpthread.so.0 => /lib/aarch64-linux-gnu/libpthread.so.0 (0x0000007fad91e000)
	libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1 (0x0000007fad8fa000)
	libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (0x0000007fad7a0000)
	libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (0x0000007fad6e7000)
	/lib/ld-linux-aarch64.so.1 (0x0000007fadaca000)
```

This `libtvm.so` is in `~/tvm/build`. Either copy this .so file to `/usr/local/lib` or using `LD_LIBRARY_PATH` environment variable to add it to library search path.

Run on agx xavier:

```
$ LD_LIBRARY_PATH=~/tvm/build RUST_BACKTRACE=1 cargo run --release ../darknet/data/dog.jpg
    Finished release [optimized] target(s) in 0.12s
     Running `target/release/yolov3-rs ../darknet/data/dog.jpg`
input size is [1, 3, 416, 416]
TVM run in 1.236270534s
15 result before nms sort
x:164 y:112 w:400 h:331 [1 0.9958774],
x:127 y:222 w:187 h:316 [16 0.99073726],
x:472 y:85 w:216 h:85 [7 0.9385087],
```

Run on 10900k:

```
input size is [1, 3, 416, 416]
TVM run in 273.787502ms
15 result before nms sort
x:164 y:112 w:400 h:331 [1 0.9958774],
x:127 y:222 w:187 h:316 [16 0.99073726],
x:472 y:85 w:216 h:85 [7 0.9385088],
```

## run on GPU

Change `target = 'llvm'` to `target = 'cuda -libs=cudnn'` in python script and run it.

Change `DeviceType::CPU` to `DeviceType::GPU` in rust code. Keep the `DeviceType` of `get_output`
to CPU because we need access output in main RAM not VRAM.

Run on agx xavier:

```
input size is [1, 3, 416, 416]
TVM run in 22.744051ms
15 result before nms sort
x:164 y:112 w:400 h:331 [1 0.9958774],
x:127 y:222 w:187 h:316 [16 0.99073726],
x:472 y:85 w:216 h:85 [7 0.9385089],
```

Run on 2080ti:

```
input size is [1, 3, 416, 416]
TVM run in 10.635469ms
15 result before nms sort
x:164 y:112 w:400 h:331 [1 0.9958774],
x:127 y:222 w:187 h:316 [16 0.99073726],
x:472 y:85 w:216 h:85 [7 0.9385089],
```

## using new module interface

You may notice this warning:

```
DeprecationWarning: legacy graph runtime behaviour of producing json / lib / params will be  removed in the next release. Please see documents of tvm.contrib.graph_runtime.GraphModule for the  new recommended usage.
  graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)
```

Change python script:

```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

lib.export_library('deploy_lib.so')
```

We will get a single huge `deploy_lib.so`

Change rust code:

```rust
let lib = Module::load(&Path::new("deploy_lib.so")).unwrap();
let default = lib.get_function("default", true).unwrap();
let ctx = Context::gpu(0);
let ret = default.invoke(vec![ctx.into()]);
let gmod: Module = ret.unwrap().try_into().unwrap();

// get the set_input function
let ref set_input_fn = gmod.get_function("set_input", false).unwrap();
...
```

## using yolov4

Same as yolov3, we need to checkout a fork version, master branch don't work.

```shell
git clone https://github.com/12101111/darknet.git
cd darknet
./build.sh
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -o weights/yolov4.weights
```

We also need to checkout a fork version of TVM.

```shell
cd ~/tvm
git remote add yolov4 https://github.com/12101111/incubator-tvm.git
git fetch yolov4
git checkout yolov4
```

Then compile yolov4 using TVM.

```python
#!/usr/bin/env python3

import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing.darknet import __darknetffi__

cfg_path = './cfg/yolov4.cfg'
weights_path = './weights/yolov4.weights'
lib_path = './libdark.so'

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network_custom(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0, 1)
dtype = 'float32'
batch_size = 1
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)
target = 'cuda --libs=cudnn'
target_host = 'llvm'

print("Compiling the model...")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

# save the model artifacts
lib.export_library('libyolov4.so')
```

The input and output shape of `yolov4` is different with `yolov3`, we need to modify it in rust.

```rust
const WIDTH: u32 = 608;
const HEIGHT: u32 = 608;

let output_shape: SmallVec<[usize; 4]> = match i % 4 {
    0 => smallvec![
        1,
        255,
        HEIGHT as usize >> (5 - (i / 4)),
        WIDTH as usize >> (5 - (i / 4))
    ,
    1 => smallvec![3],
    2 => smallvec![18],
    3 => smallvec![6],
    _ => unreachable!(),
};
```

Run on agx xavier:

```
input size is [1, 3, 608, 608]
TVM run in 64.129044ms
19 result before nms sort
x:131 y:225 w:179 h:315 [16 0.9820385],
x:121 y:122 w:448 h:302 [1 0.9248689],
x:466 y:75 w:216 h:95 [7 0.9171608],
```

Run on 2080ti:

```
input size is [1, 3, 608, 608]
TVM run in 24.377104ms
19 result before nms sort
x:131 y:225 w:179 h:315 [16 0.9820385],
x:121 y:122 w:448 h:302 [1 0.9248689],
x:466 y:75 w:216 h:95 [7 0.9171608],
```
