# Step 1: Export and Quantize the Model

## Step 1-1: Export the model

Here we take ResNet50 as an example. Please run the following command to export `resnet50.onnx`.

```
python export.py
```

## Step 1-2: Pre-process the model

Pre-processing prepares a float32 model for quantization. Run the following command to pre-process model `resnet50.onnx` to `resnet50-infer.onnx`.

```
python -m onnxruntime.quantization.preprocess --input resnet50.onnx --output resnet50-infer.onnx
```

Quantization requires tensor shape information to perform its best. Model optimization also improve the performance of quantization. For instance, a Convolution node followed by a BatchNormalization node can be merged into a single node during optimization. Currently we can not quantize BatchNormalization by itself, but we can quantize the merged Convolution + BatchNormalization node. It is highly recommended to run model optimization in pre-processing instead of in quantization.

## Step 1-3: Quantize the model

Quantization tool takes the pre-processed float32 model and produce a quantized model. It's recommended to use Tensor-oriented quantization (QDQ; Quantize and DeQuantize).

I have prepared [test_images.tar.gz](https://github.com/chenyaofo/deep-learning-model-deploy-tutorial/releases/download/v0.1/test_images.tar.gz), please download uncompress it.

```
python quantize.py --input_model resnet50-infer.onnx --output_model resnet50-quant.onnx --calibrate_dataset test_images
```

## Step 1-4: Check the Differences between FLoat and Quantized Model

Quantization is not a loss-less process. Sometime it results in significant loss in accuracy. To help locate the source of these losses, our quantization debugging tool matches up weight tensors of the float32 model vs those of the quantized model. If a input data reader is provided, our debugger can also run both models with the same input and compare their corresponding tensors:

```
python check.py --float_model resnet50-infer.onnx --qdq_model resnet50-quant.onnx  --calibrate_dataset test_images
```

## Reference

https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md