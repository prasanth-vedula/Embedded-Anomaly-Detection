# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2026-08-16

### Added
- Deterministic synthetic five-sensor dataset generation.
- Canonical preprocessing and 16-sample windowing.
- Dense autoencoder with `80 → 32 → 8 → 32 → 80` architecture.
- Healthy-validation threshold calibration at the 99th percentile.
- Four explicit fault scenarios.
- Deterministic INT8 quantization and verification.
- C deployment export for embedded inference.
- Embedded anomaly state machine with persistence confirmation.
- Structured anomaly telemetry.
- Python, integration and embedded C test suites.
- Scenario example scripts and documentation.
- GitHub Actions CI workflow.
- Release-oriented repository documentation.

### Verification
- 25 Python/integration tests passing.
- 3 embedded C tests passing.
- Deployment-consistency tests passing.
