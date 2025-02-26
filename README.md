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

0. make_key
```
cd key
gcc -o make_key make_key.c
./make_key [key_size] [key_filename]
```

1. encrypt
```
cd server/Encrypto
mkdir build && cd build
cmake ..
make
./encrypto [model_file_path] [out_file_name] [key_file_path]
```