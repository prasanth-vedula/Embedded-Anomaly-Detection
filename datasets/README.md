# Datasets

The project uses deterministic synthetic five-sensor data for reproducible TinyML development and testing.

## Sensors

1. Temperature
2. Vibration
3. Pressure
4. Current
5. Humidity

The dataset generator produces healthy operation plus explicit fault scenarios including temperature drift, vibration disturbance, current disturbance and multi-sensor disturbance.

The repository does not claim that these synthetic distributions represent field measurements. They are used to validate the complete training-to-deployment pipeline deterministically.

Generated data can be recreated through the Python data-generation module rather than committing large generated datasets to the repository.
