# Multi-Sensor Fault Example

Demonstrates the strongest synthetic disturbance, where multiple sensor channels deviate simultaneously.

Run:

```powershell
python ".\examples\multi_sensor_fault\run_example.py"
```

Expected behavior: simultaneous disturbances produce a strong reconstruction-error response.

Current reference run:
- Threshold: `0.19294606685638452`
- Mean error: `2.0771849155426025`
- P95 error: `4.897256851196289`
- Anomaly rate: `50.31%`
