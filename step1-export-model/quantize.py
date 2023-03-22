"""
Modified from https://raw.githubusercontent.com/microsoft/onnxruntime-inference-examples/main/quantization/image_classification/cpu/run.py
"""

import os
import time
import argparse

import numpy as np
import onnxruntime
from PIL import Image

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization import CalibrationDataReader



def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    # image_names = os.listdir(images_folder)
    # if size_limit > 0 and len(image_names) >= size_limit:
    #     batch_filenames = [image_names[i] for i in range(size_limit)]
    # else:
    #     batch_filenames = image_names
    # unconcatenated_batch_data = []

    # for image_name in batch_filenames:
    #     image_filepath = images_folder + "/" + image_name
    #     pillow_img = Image.new("RGB", (width, height))
    #     pillow_img.paste(Image.open(image_filepath).resize((width, height)))
    #     input_data = np.float32(pillow_img) - np.array(
    #         [123.68, 116.78, 103.94], dtype=np.float32
    #     )
    #     nhwc_data = np.expand_dims(input_data, axis=0)
    #     nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
    #     unconcatenated_batch_data.append(nchw_data)
    # batch_data = np.concatenate(
    #     np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    # )
    
    # import ipdb; ipdb.set_trace()

    import torch
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    # std=[1, 1, 1],
                                    )

    val_dataset = datasets.ImageFolder(
            images_folder,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, sampler=None)  
    for images, _ in val_loader:
        images: torch.Tensor
        batch_data = images.unsqueeze(1).numpy()
    return batch_data


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", default="./test_images", help="calibration data set"
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=True, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    dr = ResNet50DataReader(
        calibration_dataset_path, input_model_path
    )

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, create_calibrator
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        optimize_model=False,
        calibrate_method=CalibrationMethod.Entropy,
        reduce_range=False
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()