# Vibration Fault Example

Demonstrates detection of abnormal vibration behavior.

Run:

```powershell
python ".\examples\vibration_fault\run_example.py"
```

Expected behavior: vibration disturbance increases reconstruction error and produces a substantially higher anomaly rate.

Current reference run:
- Threshold: `0.19294606685638452`
- Mean error: `1.3142993450164795`
- P95 error: `3.4942548274993896`
- Anomaly rate: `50.31%`
