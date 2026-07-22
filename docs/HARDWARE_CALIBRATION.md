# Hardware Calibration

`hardware inspect` inventories CPU, Torch, CUDA, devices, precision support, and backend capabilities. `hardware estimate` is theoretical. `hardware calibrate` executes bounded probes and reports the largest safe configuration observed under the requested memory fraction.

Calibration never silently mutates a scientific run. Copy the recommended values into a new resolved manifest and record the change. CUDA allocator peaks, reserved memory, selected precision, and failures must remain visible.

The v0.6 build environment was CPU-only; 8 GB CUDA presets require target-machine validation.
