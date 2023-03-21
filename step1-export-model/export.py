import torch
import torchvision.models as models
import onnx
from onnxsim import simplify

batch_size = 1
EXPORT_MODEL_NAME ="resnet50.onnx"

model = models.resnet50(pretrained=True)
model.eval()

x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
y = model(x)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  EXPORT_MODEL_NAME,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

model = onnx.load(EXPORT_MODEL_NAME)
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

onnx.save(model, EXPORT_MODEL_NAME)

onnx_model = onnx.load(EXPORT_MODEL_NAME)
onnx.checker.check_model(onnx_model)