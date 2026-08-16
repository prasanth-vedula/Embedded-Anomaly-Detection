# Current Fault Example

Demonstrates detection of abnormal current behavior.

Run:

```powershell
python ".\examples\current_fault\run_example.py"
```

Expected behavior: current disturbance increases reconstruction error relative to normal operation.

Current reference run:
- Threshold: `0.19294606685638452`
- Mean error: `0.39116212725639343`
- P95 error: `1.215505599975586`
- Anomaly rate: `28.97%`
