# AIoT Security : model info encrypt-decrypt using NPU
- encrypt contents : layer(node) of model configuration info, weight
- decrypt : use npu

## Requirements
- probobuf-c
- onnx.proto
```
 $ sudo apt install protobuf-compiler libprotobuf-c-dev protobuf-c-compiler
```

## usage

1. encrypt
```
cd server/Encrypto
mkdir build && cd build
cmake ..
make
```