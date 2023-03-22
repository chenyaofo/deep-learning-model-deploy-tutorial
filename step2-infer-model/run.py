import numpy
import onnxruntime
import onnxruntime.backend as backend

MODEL = "resnet50-quant.onnx"
# MODEL = "resnet50-infer.onnx"

rep = backend.prepare(MODEL, onnxruntime.get_device())

predictions = rep.run(numpy.random.rand(10,3,224,224).astype(numpy.float32))

print(
    f"Infer model={rep._session._model_path}, "
    f"provider={rep._session._providers}, ",
    f"predictions={predictions[0].argmax(-1)}"
)
