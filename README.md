# Embedded Anomaly Detection — TinyML Edge Inference

> **A production-style embedded TinyML anomaly-detection pipeline that trains a dense autoencoder in Python, calibrates an anomaly threshold from healthy validation data, quantizes the model to INT8, exports the deployment representation to C, and executes deterministic anomaly inference in an embedded-style C runtime.**

![Language](https://img.shields.io/badge/Python-3.11-blue)
![Embedded](https://img.shields.io/badge/Embedded-C99-orange)
![TinyML](https://img.shields.io/badge/TinyML-INT8-success)
![Build](https://img.shields.io/badge/Build-CMake%20%2B%20Ninja-blue)
![Tests](https://img.shields.io/badge/Tests-25%20Python%20%2B%203%20Embedded-success)
![Model](https://img.shields.io/badge/Model-80--32--8--32--80-informational)
![Parameters](https://img.shields.io/badge/Parameters-5%2C784-informational)
![Status](https://img.shields.io/badge/Status-v1.0.0%20Release-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

**Embedded Anomaly Detection** is an end-to-end TinyML/embedded-systems project for detecting abnormal industrial sensor behavior at the edge.

The project combines a Python machine-learning pipeline with a C deployment runtime. Healthy sensor data is used to train a compact dense autoencoder. The trained model learns the normal operating manifold and detects abnormal behavior through reconstruction error.

The deployment path is deliberately explicit:

```text
Sensor data
    ↓
Validation
    ↓
Canonical preprocessing
    ↓
16-sample sliding window
    ↓
80-value model input
    ↓
Dense autoencoder
80 → 32 → 8 → 32 → 80
    ↓
Reconstruction error
    ↓
Calibrated threshold
    ↓
Persistence / confirmation logic
    ↓
Anomaly event
    ↓
Telemetry
```

The same model definition, preprocessing dimensions, threshold and deployment layers are verified across the Python and embedded artifacts.

---

## Why this project matters

This repository is designed to demonstrate more than a machine-learning model.

It demonstrates the engineering work required to move a TinyML algorithm toward an embedded deployment:

- deterministic dataset generation
- canonical preprocessing
- reproducible training
- healthy-only threshold calibration
- model serialization
- INT8 quantization
- quantization verification
- C source/header generation
- embedded inference
- anomaly persistence logic
- structured telemetry
- CMake builds
- Python and C tests
- deployment-consistency tests
- GitHub Actions CI
- release-oriented repository structure

The goal is a **traceable ML-to-firmware pipeline**, rather than an isolated notebook experiment.

---

# Key Specifications

| Component | Implementation |
|---|---|
| Model | Dense Autoencoder |
| Architecture | `80 → 32 → 8 → 32 → 80` |
| Latent size | 8 |
| Parameters | 5,784 |
| Input window | 16 samples |
| Sensor features | 5 |
| Model input | 80 values |
| Hidden activation | ReLU |
| Output activation | Linear |
| Deployment | INT8 weights + C runtime |
| Threshold | `0.1929460669` |
| Threshold method | Healthy validation 99th percentile |
| Confirmation count | 3 consecutive anomalous scores |
| Embedded language | C99 |
| Host/ML language | Python |
| Build | CMake + Ninja |
| Testing | pytest + CTest |
| CI | GitHub Actions |

---

# Supported Detection Scenarios

The evaluation pipeline includes four explicit fault scenarios plus normal operation.

| Scenario | Mean reconstruction error | Anomaly rate |
|---|---:|---:|
| Normal operation | `0.11047` | `1.38%` |
| Temperature drift | `1.47157` | `45.66%` |
| Vibration fault | `1.31430` | `50.31%` |
| Current fault | `0.39116` | `28.97%` |
| Multi-sensor fault | `2.07718` | `50.31%` |

These figures come from the current deterministic evaluation run and are included as project evidence, not as a claim of field accuracy.

---

# Threshold Calibration

The anomaly threshold is **not hard-coded from an arbitrary guess**.

It is calibrated from healthy validation reconstruction errors:

```text
Healthy validation samples: 297
Quantile:                    0.99
Threshold:                   0.19294606685638452
Mean validation error:       0.1217134619752566
P95 validation error:        0.17219465970993042
P99 validation error:        0.19294606685638452
Maximum validation error:    0.20974144339561462
```

This creates a reproducible relationship between model training, validation data and deployment threshold.

---

# Model and Deployment Pipeline

```text
                    PYTHON / HOST SIDE
┌─────────────────────────────────────────────────────────────┐
│ Deterministic sensor dataset                               │
│          ↓                                                  │
│ Canonical preprocessing                                    │
│          ↓                                                  │
│ Sliding windows: [16, 5]                                   │
│          ↓                                                  │
│ Flatten: [80]                                               │
│          ↓                                                  │
│ Dense Autoencoder                                           │
│ 80 → 32 → 8 → 32 → 80                                      │
│          ↓                                                  │
│ Healthy validation                                          │
│          ↓                                                  │
│ Threshold calibration                                      │
│          ↓                                                  │
│ INT8 quantization                                           │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
                    DEPLOYMENT ARTIFACTS
┌─────────────────────────────────────────────────────────────┐
│ anomaly_model.json                                         │
│ anomaly_model_int8.json                                    │
│ model_metadata.json                                        │
│ preprocessing_parameters.json                              │
│ model_weights.h / model_weights.c                          │
│ deployment_parameters.h / deployment_parameters.c          │
└─────────────────────────────┬───────────────────────────────┘
                              ↓
                    EMBEDDED C RUNTIME
┌─────────────────────────────────────────────────────────────┐
│ Sensor simulation / input                                  │
│          ↓                                                  │
│ Validation + preprocessing                                 │
│          ↓                                                  │
│ 16-sample window                                            │
│          ↓                                                  │
│ INT8 model inference                                       │
│          ↓                                                  │
│ Reconstruction score                                       │
│          ↓                                                  │
│ Persistence / confirmation                                 │
│          ↓                                                  │
│ Fault mask + event                                         │
│          ↓                                                  │
│ JSON telemetry                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# Embedded Runtime

The C side is organized into focused modules:

```text
embedded/
├── Inc/
│   ├── Anomaly/
│   ├── Common/
│   ├── Config/
│   ├── Event/
│   ├── HAL/
│   ├── Preprocess/
│   ├── Sensor/
│   ├── Telemetry/
│   ├── TinyML/
│   └── Window/
│
└── Src/
    ├── Anomaly/
    ├── Event/
    ├── HAL/
    ├── Preprocess/
    ├── Sensor/
    ├── Telemetry/
    ├── TinyML/
    ├── Window/
    └── main.c
```

### Runtime responsibilities

**Sensor layer**
- validates sensor samples
- provides deterministic simulation data

**Preprocessing**
- converts raw sensor measurements into the deployment feature representation

**Windowing**
- maintains the 16-sample sliding input window

**TinyML**
- executes the exported compact autoencoder

**Anomaly logic**
- compares reconstruction error with the calibrated threshold
- requires persistence before confirmation

**Event layer**
- converts confirmed anomalies into structured events

**Telemetry**
- emits machine-readable JSON records

---

# Example Telemetry

A confirmed embedded anomaly produces records such as:

```json
{
  "ts": 19600,
  "score": 2.490719,
  "severity": 2,
  "fault_mask": 31
}
```

Where:

- `ts` = simulated timestamp in milliseconds
- `score` = reconstruction error
- `severity` = anomaly state
- `fault_mask` = detected sensor/fault bit mask

The runtime demonstrates persistent anomaly confirmation rather than triggering an event from a single noisy sample.

---

# INT8 Quantization

The deployment model is exported using deterministic INT8 quantization.

The repository verifies:

- quantized tensor structure
- tensor shapes
- INT8 ranges
- quantization scales
- dequantized inference consistency
- anomaly-score consistency
- threshold preservation
- parameter-count preservation

The exported model retains the same topology:

```text
80 → 32 → 8 → 32 → 80
```

with all four deployment layers:

```text
encoder
latent
decoder
output
```

---

# Verification

The current repository contains **25 Python/integration tests** and **3 embedded C tests**.

## Python + integration

```text
25 passed
```

Coverage includes:

- dataset reproducibility
- preprocessing
- window generation
- scaler round-trip
- model architecture
- model training
- model reload consistency
- threshold calibration
- scenario evaluation
- INT8 structure
- quantization ranges
- dequantized inference
- anomaly-score preservation
- deployment metadata consistency
- exported C header consistency
- end-to-end artifact chain

## Embedded

```text
3/3 tests passed
```

The embedded suite verifies:

- core embedded behavior
- preprocessing behavior
- TinyML inference behavior

---

# Quick Start

## Requirements

- Python 3.11+
- GCC / MinGW or another C99 compiler
- CMake 3.16+
- Ninja
- pytest
- NumPy

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## Run Python tests

```powershell
python -m pytest tests\python tests\integration -v
```

Expected:

```text
25 passed
```

---

## Train the model

```powershell
python -m python.training.train_model
```

This generates the FP32 model artifact and training metadata.

---

## Calibrate and evaluate

```powershell
python -m python.evaluation.evaluate
```

This calibrates the anomaly threshold from healthy validation data and evaluates all scenarios.

---

## Quantize

```powershell
python -m python.quantization.quantize_model
```

This generates:

```text
models/anomaly_model_int8.json
```

---

## Export to C

```powershell
python -m python.export.export_model
```

This generates the embedded deployment artifacts:

```text
embedded/Inc/TinyML/model_weights.h
embedded/Src/TinyML/model_weights.c
embedded/Inc/Config/deployment_parameters.h
embedded/Src/Config/deployment_parameters.c
```

---

## Build the embedded runtime

```powershell
cmake -S .\embedded -B .\build\embedded
cmake --build .\build\embedded --config Release
```

---

## Run embedded tests

```powershell
ctest --test-dir ".\build\embedded" --output-on-failure
```

Expected:

```text
100% tests passed
```

---

## Run the embedded demonstration

```powershell
.\build\embedded\embedded_anomaly_detection.exe
```

The application emits structured anomaly telemetry.

---

# Scenario Examples

Each scenario has an independently runnable demonstration:

```text
examples/
├── normal_operation/
├── vibration_fault/
├── current_fault/
└── multi_sensor_fault/
```

Run them with:

```powershell
python ".\examples\normal_operation\run_example.py"
python ".\examples\vibration_fault\run_example.py"
python ".\examples\current_fault\run_example.py"
python ".\examples\multi_sensor_fault\run_example.py"
```

Each example reports the model architecture, threshold, reconstruction-error statistics and anomaly rate.

---

# Repository Structure

```text
Embedded-Anomaly-Detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── datasets/
│   └── README.md
│
├── embedded/
│   ├── Inc/
│   ├── Src/
│   └── CMakeLists.txt
│
├── examples/
│   ├── normal_operation/
│   ├── vibration_fault/
│   ├── current_fault/
│   └── multi_sensor_fault/
│
├── models/
│   ├── anomaly_model.json
│   ├── anomaly_model_int8.json
│   ├── model_metadata.json
│   └── preprocessing_parameters.json
│
├── python/
│   ├── data/
│   ├── evaluation/
│   ├── export/
│   ├── preprocessing/
│   ├── quantization/
│   ├── training/
│   └── visualization/
│
├── tests/
│   ├── python/
│   ├── integration/
│   └── embedded/
│
├── screenshots/
│
├── CMakeLists.txt
├── pyproject.toml
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
├── SECURITY.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── PROJECT_INFO.md
└── README.md
```

---

# Engineering Decisions

## Canonical preprocessing

Training, evaluation and deployment use the same preprocessing contract. This prevents silent feature-scaling differences between the Python model and embedded runtime.

## Healthy-only calibration

The anomaly threshold is derived from healthy validation reconstruction errors rather than from the fault scenarios.

## Deterministic training

The pipeline uses a fixed seed and explicit training configuration so model generation can be reproduced.

## Compact architecture

The 8-dimensional latent representation keeps the model small enough for embedded experimentation while retaining a useful reconstruction-based anomaly signal.

## INT8 deployment representation

Quantized weights reduce the representation size and make the deployment artifact more appropriate for resource-constrained inference.

## Generated deployment source

The C model representation is generated from the verified quantized artifact instead of being manually copied into firmware.

## Persistence-based confirmation

A score crossing the threshold does not immediately become a confirmed event. The embedded state machine requires consecutive anomalous observations, reducing sensitivity to isolated spikes.

---

# Testing Philosophy

The repository uses testing at three levels:

```text
Unit tests
    ↓
Integration / artifact consistency
    ↓
Embedded runtime tests
```

This gives the project a traceable verification chain:

```text
Python model
     ↓
Quantized artifact
     ↓
C export
     ↓
Embedded inference
     ↓
Telemetry
```

A change that breaks the artifact contract should be caught before deployment.

---

# Evidence and Reproducibility

The `screenshots/` directory contains selected terminal logs captured during verification.

Raw text logs are intentionally retained because they preserve exact command output.

Recommended visual screenshots should focus on:

1. model/training summary
2. threshold calibration
3. INT8 quantization
4. 25 Python tests passing
5. 3 embedded tests passing
6. deployment consistency
7. production telemetry

---

# Limitations

This project is a **simulation and embedded-runtime demonstration**, not a validated industrial product.

In particular:

- sensor inputs are simulated
- fault distributions are synthetic
- no field deployment accuracy is claimed
- no MCU-specific benchmark is claimed
- latency and RAM/flash measurements are not presented as hardware measurements
- the current embedded target is a host-built C runtime rather than a specific production MCU board

These boundaries are intentional so that the repository does not overstate its validation level.

---

# Future Work

Potential next steps include:

- STM32 hardware deployment
- CMSIS-NN optimized kernels
- cycle-count benchmarking
- RAM/Flash measurement on target hardware
- real industrial sensor acquisition
- hardware-in-the-loop testing
- additional fault classes
- online model monitoring
- DMA-based acquisition
- FreeRTOS integration
- MCU-specific power measurements

---

# Release

Current release target:

**v1.0.0 — Initial stable embedded TinyML pipeline**

The release represents the completed host-side training/evaluation pipeline, deterministic INT8 deployment artifact, C export, embedded runtime, examples and verification suite.

---

# Author

**Vedula China Venkata Prasanth**

B.Tech — Electronics and Communication Engineering
Embedded Systems • Firmware • TinyML • Edge AI • C/C++ • Python

Vizianagaram, Andhra Pradesh, India

---

# License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Project Status

| Area | Status |
|---|---|
| Dataset generation | ✅ Complete |
| Preprocessing | ✅ Complete |
| Autoencoder training | ✅ Complete |
| Threshold calibration | ✅ Complete |
| Fault evaluation | ✅ Complete |
| INT8 quantization | ✅ Complete |
| C export | ✅ Complete |
| Embedded runtime | ✅ Complete |
| Embedded tests | ✅ Complete |
| Integration tests | ✅ Complete |
| Examples | ✅ Complete |
| CI | ✅ Configured |
| Documentation | 🟢 Release polish |
| Version | `v1.0.0` |

---

⭐ If this project is useful, consider starring the repository.
