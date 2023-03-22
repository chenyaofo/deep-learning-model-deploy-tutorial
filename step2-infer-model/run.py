import numpy
import torch
import onnxruntime
import onnxruntime.backend as backend

from data import val_loader
MODEL = "resnet50-quant.onnx"
# MODEL = "resnet50-infer.onnx"

fp32_rep = backend.prepare("resnet50-infer.onnx", onnxruntime.get_device())
quant_rep = backend.prepare("resnet50-quant.onnx", onnxruntime.get_device())


print(
    f"Infer model={fp32_rep._session._model_path}, "
    f"provider={fp32_rep._session._providers}, ",
)

for images, targets in val_loader:
    images: torch.Tensor
    # images = images.permute(0,2,3,1)
    fp32_rev = fp32_rep.run(images.numpy())
    quant_rev = quant_rep.run(images.numpy())

    fp32_classes = fp32_rev[0].argmax(-1)
    quant_classes = quant_rev[0].argmax(-1)

    print(
        f"fp32_rev={fp32_classes}, "
        f"quant_rev={quant_classes}, "
        f"#same={(fp32_classes==quant_classes).astype(numpy.int32).sum()}/256"
    )