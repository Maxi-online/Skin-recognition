# Third-party notices

This release contains project source and project-trained projection parameters. It does not bundle dependency distributions, face datasets, or third-party ONNX model files. Installing dependencies or obtaining models creates separate license obligations; the project's research license does not replace them.

## InsightFace and pretrained models

The project uses **InsightFace 0.7.3** through its Python API and ONNX model interfaces. The [upstream Python library](https://github.com/deepinsight/insightface/tree/master/python-package) is MIT-licensed. Upstream distinguishes the library code from its supplied pretrained models: those models are restricted to **noncommercial research**, including manually downloaded models. Review the current [upstream license statement](https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/README.md) and [model distribution](https://github.com/deepinsight/insightface/releases/tag/v0.7) before use.

The frozen configuration references SCRFD detection (`det_10g.onnx`), 106-point landmarks (`2d106det.onnx`), and the WebFace600K-trained ArcFace R50 model (`w600k_r50.onnx`). These files are not included. Their expected SHA-256 values are recorded in the profiles. A license for this project's code does not authorize commercial use of these weights.

## Software dependencies

The dependency lists pin the tested versions. The following are upstream license references, not a replacement for the license files supplied with installed distributions and their bundled components:

| Component | Upstream license information |
|---|---|
| NumPy | [BSD 3-Clause and bundled component notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy | [BSD 3-Clause and bundled component notices](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| scikit-learn | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| scikit-image | [BSD-family and component notices](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt) |
| OpenCV | [Apache 2.0 and third-party notices](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| Pillow | [HPND and bundled component notices](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| Matplotlib | [Matplotlib license and historical notices](https://matplotlib.org/stable/project/license.html) |
| ONNX Runtime | [MIT and third-party notices](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| PyTorch / torchvision | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| FastAPI / Uvicorn / Pydantic | [FastAPI MIT](https://github.com/fastapi/fastapi/blob/master/LICENSE), [Uvicorn BSD](https://github.com/encode/uvicorn/blob/main/LICENSE.md), [Pydantic MIT](https://github.com/pydantic/pydantic/blob/main/LICENSE) |
| Dash / dash-bootstrap-components | [Dash MIT](https://github.com/plotly/dash/blob/dev/LICENSE), [dash-bootstrap-components Apache 2.0](https://github.com/dbc-team/dash-bootstrap-components/blob/main/LICENSE) |
| pytest / HTTPX | [pytest MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE), [HTTPX BSD](https://github.com/encode/httpx/blob/master/LICENSE.md) |

Other transitive or optional dependencies remain subject to the notices in their distributions. CUDA, cuDNN, and other NVIDIA runtime components have separate vendor terms.

## Research data

Acquisition utilities refer to FEI, CFP, Georgia Tech, LFW, and ORL/AT&T research resources. Dataset ownership, research-only restrictions, participant permissions, and redistribution terms remain with their respective providers. No dataset photographs or per-subject biometric templates are included. Downloading, redistributing, or using a dataset must be evaluated under its own terms; availability of an acquisition script is not a license grant.

## Project-trained parameters

`research/development/v4_classical_multicondition/models/pure_skin_multicondition_best.npz` contains the project's learned PCA and within-identity whitening parameters, trained on the FEI development partition. It contains no input images or per-sample embedding table. These parameters are distributed for noncommercial research under the project license; this does not transfer rights in the underlying dataset.
