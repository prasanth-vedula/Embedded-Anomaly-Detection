# Project Information

## Project

**Embedded Anomaly Detection — TinyML Edge Inference**

A deterministic embedded anomaly-detection pipeline that trains a compact dense autoencoder,
calibrates an anomaly threshold from healthy validation data, quantizes the deployment model to
INT8, exports C model parameters, and executes inference in a standalone embedded C runtime.

## Engineering classification

- Domain: Embedded Systems / TinyML / Edge AI
- Primary languages: C, Python
- ML approach: Dense autoencoder
- Deployment representation: INT8 quantized weights with per-layer scales
- Embedded build: CMake + Ninja + GCC
- Verification: pytest + CTest
- Development target: portable host-side embedded simulation/runtime

## Model

```text
80 → 32 → 8 → 32 → 80
```

- Input size: 80
- Latent size: 8
- Parameter count: 5,784
- Hidden activation: ReLU
- Output activation: linear
- Training seed: 42
- Training epochs: 250
- Batch size: 64
- Learning rate: 0.001

## Detection

The anomaly threshold is calibrated from healthy validation reconstruction errors using the 0.99
quantile. The calibrated threshold is propagated through the FP32 metadata, INT8 deployment artifact,
and generated embedded C header.

## Embedded pipeline

```text
Sensor simulation
      ↓
Validation
      ↓
Preprocessing
      ↓
16-sample window
      ↓
80-element model input
      ↓
INT8 TinyML inference
      ↓
Reconstruction error
      ↓
Threshold + persistence
      ↓
Anomaly event
      ↓
Telemetry
```

## Verification baseline

The finalized project currently verifies:

- Python unit/integration suite: 25 tests passing.
- Embedded CTest suite: 3 tests passing.
- Deployment consistency suite: 5 tests passing.
- FP32/INT8 architecture consistency.
- Threshold propagation consistency.
- Exported embedded header consistency.
- Normal, vibration-fault, current-fault, and multi-sensor-fault demonstrations.

## Scope and limitations

This repository is a portfolio/research-grade engineering implementation and simulation. It is not
represented as a certified industrial safety system. Sensor models, fault models, timing behavior,
and deployment constraints should be revalidated for a real MCU and real plant before production use.

## Versioning

Releases should use semantic versioning where practical. Generated model artifacts must be regenerated
and verified whenever model architecture, preprocessing, or threshold calibration changes.
