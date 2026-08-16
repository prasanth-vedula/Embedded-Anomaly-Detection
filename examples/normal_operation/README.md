# Normal Operation Example

Demonstrates the baseline behavior of the autoencoder on healthy sensor data.

Run:

```powershell
python ".\examples\normal_operation\run_example.py"
```

Expected behavior: reconstruction error remains predominantly below the calibrated anomaly threshold.

Current reference run:
- Threshold: `0.19294606685638452`
- Mean error: `0.11047397553920746`
- P95 error: `0.1698564738035202`
- Anomaly rate: `1.38%`
