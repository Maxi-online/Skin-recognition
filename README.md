# Skin Recognition

A research implementation of quality-controlled facial skin texture, whole-face embeddings, and their fusion. Built by **Maxim Strekolovsky / AERIS LLC**.

The project connects reproducible computer-vision experiments to a local FastAPI service and a Dash interface. It keeps image-quality rejection, matching scores, and gallery transactions explicit. The application supports research enrollment and matching; scores express normalized similarity. Liveness and presentation-attack detection are separate system requirements.

## Methods

| Profile | Representation | Dimension |
|---|---|---:|
| `skin_texture` | Dense RootSIFT on landmark-guided skin patches; training-only PCA and within-identity whitening | 256 |
| `whole_face` | ArcFace R50, with the same strict quality gate as the skin branch | 512 |
| `hybrid` | Weighted concatenation of normalized face and skin embeddings; skin weight 0.02 | 768 |
| `whole_face_detector_only` | ArcFace R50 with detector and target-selection checks | 512 |

Version 4.2 preserves native image scale for small inputs through centered padding. When multiple faces are detected, it accepts a target only when exactly one bounding box contains the frame center. Skin extraction requires at least two usable patches. The representation describes 2D facial appearance and local skin texture.

On the held-out FEI protocol, the skin-texture profile achieved **97.22% operational rank-1** (350/360 probes); whole-face and hybrid profiles both achieved **98.33%** (354/360 probes). Quality failures count as failures. On the independent CFP frontal-frontal protocol, the whole-face profile achieved **96.41% all-pair accuracy** (7,000 pairs); the hybrid result was 96.39%. The profiles therefore tied on FEI, while whole-face embeddings led slightly on CFP. Performance depends on the dataset, acquisition conditions, quality coverage, and frozen operating point.

## Included

- Inference, preprocessing, descriptors, transactional NPZ gallery, API, and UI source.
- Training, calibration, protocol, extraction, and evaluation scripts.
- Unit and integration tests, pinned dependency lists, and Windows launcher.
- The project's trained PCA/whitening model, containing projection parameters rather than per-person embeddings.
- Portable v4.1/v4.2 inference profiles with source/model hashes and unchanged numerical parameters.

Face images, individual embeddings, enrollment galleries, external ONNX weights, and manuscripts are not distributed. The profiles retain references and hashes for historical experiment protocols and development reports; those optional audit inputs are excluded from this code release. Inference uses the included code and projection plus the separately obtained ONNX models. Running the experiment scripts requires obtaining the datasets independently and preparing their protocol files.

## Run locally on Windows

The frozen v4.2 profile uses NVIDIA CUDA. The verified development environment used Python 3.13, PyTorch 2.9.1 with CUDA 12.8, and ONNX Runtime GPU 1.23.2. Use the pinned versions: changing the runtime, models, preprocessing, or projection invalidates the original calibration contract.

From the repository root, create the GPU environment:

```powershell
py -3.13 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

InsightFace 0.7.3 may require C++ build tools when a compatible wheel is unavailable. Do not install the CPU and GPU ONNX Runtime distributions into the same environment.

Obtain the `buffalo_l` models from the [official InsightFace model distribution](https://github.com/deepinsight/insightface/releases/tag/v0.7), subject to its separate noncommercial research terms. Place these three files in `models/buffalo_l/`:

```text
models/buffalo_l/det_10g.onnx
models/buffalo_l/2d106det.onnx
models/buffalo_l/w600k_r50.onnx
```

Verify the model and frozen source hashes, then create local path-only copies of the profiles:

```powershell
.\.venv-gpu\Scripts\python.exe tools/relocate_research_profile.py --model-dir models/buffalo_l --dry-run
.\.venv-gpu\Scripts\python.exe tools/relocate_research_profile.py --model-dir models/buffalo_l
.\start-research.ps1 -Backend v42 -Method hybrid -FreezePath research/frozen/relocated_study_v4_2.json -CheckOnly
.\start-research.ps1 -Backend v42 -Method hybrid -FreezePath research/frozen/relocated_study_v4_2.json
```

The service binds to `127.0.0.1:8001` with one worker. Open the [local interface](http://127.0.0.1:8001/) or [API documentation](http://127.0.0.1:8001/docs). Enrollment creates a separate local gallery under `runtime/`. All model weights are supplied locally. Obtain consent and appropriate rights before processing photographs.

The service accepts a **full image**, performs detection/alignment itself, and uses BGR `uint8` arrays internally. Pass the original full frame to the v4.2 Python recognition service. The gallery is bound to the model, preprocessing, and profile identities; changing a profile requires a compatible new gallery.

## Tests and experiment code

CPU tests can run independently of the CUDA inference profile:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests requiring local research fixtures are skipped when those fixtures are absent. Synthetic tests verify software contracts; biometric performance is evaluated separately on research datasets.

`experiments/` contains development, frozen extraction, and evaluation entry points. `research/data/` contains dataset acquisition/audit utilities; running them is a separate action subject to each dataset's terms. Preserve disjoint training, calibration, and evaluation subjects and never tune on the final evaluation results. Historical experimental branches remain available for comparison; the default application profile is v4.2.

```text
src/                  Preprocessing, descriptors, services, API, and UI
experiments/          Training, calibration, and evaluation workflows
tests/                Software regression tests
config/               Default local application profile
research/frozen/      Portable frozen inference profiles
research/development/ Trained projection parameters
tools/                Profile relocation and integrity verification
```

## License and contact

Copyright 2026 Maxim Strekolovsky and AERIS LLC. This is **source-available software for noncommercial research**, under [LICENSE](LICENSE), not an unrestricted open-source license. Commercial or production use requires a separate written license. Dependencies and pretrained weights retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Contact: [info@aeris-aerospace.ru](mailto:info@aeris-aerospace.ru). Author identifier: [ORCID 0009-0009-7005-9490](https://orcid.org/0009-0009-7005-9490).
