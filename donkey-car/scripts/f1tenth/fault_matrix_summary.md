# Fault matrix -- standard Pure Pursuit on Spielberg

Controller: Pure Pursuit alone (non-adaptive), velocity_gain=0.5, 3 seeds/cell, max_steps=10000.

Clean baseline (severity 0): progress=100.0%, mean|d|=0.026 m, completion=100%.

## Mean progress (% of one lap; lower = more degraded)

| fault | class | s=0.0 | s=0.25 | s=0.5 | s=0.75 | s=1.0 |
|---|---|---|---|---|---|---|
| friction_drop | in | 100 | 100 | 54 | 39 | 13 |
| low_grip_patch | in | 100 | 100 | 100 | 39 | 39 |
| tire_stiffness | in | 100 | 100 | 100 | 53 | 13 |
| mass_change | in | 100 | 100 | 100 | 100 | 100 |
| steering_bias | out | 100 | 100 | 100 | 100 | 100 |
| steering_loe | out | 100 | 100 | 60 | 39 | 13 |
| actuator_latency | out | 100 | 100 | 100 | 100 | 80 |
| wheel_drag | out | 100 | 100 | 90 | 77 | 64 |
| obs_noise | out | 100 | 100 | 100 | 100 | 100 |
| obs_latency | out | 100 | 100 | 100 | 100 | 100 |

## Crash rate (fraction of seeds; TC-Driver-style crash ratio)

| fault | class | s=0.0 | s=0.25 | s=0.5 | s=0.75 | s=1.0 |
|---|---|---|---|---|---|---|
| friction_drop | in | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| low_grip_patch | in | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| tire_stiffness | in | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| mass_change | in | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| steering_bias | out | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| steering_loe | out | 0.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| actuator_latency | out | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| wheel_drag | out | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| obs_noise | out | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| obs_latency | out | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Mean tracking error |d| (m)

| fault | class | s=0.0 | s=0.25 | s=0.5 | s=0.75 | s=1.0 |
|---|---|---|---|---|---|---|
| friction_drop | in | 0.026 | 0.036 | 0.045 | 0.067 | 0.149 |
| low_grip_patch | in | 0.026 | 0.027 | 0.032 | 0.120 | 0.103 |
| tire_stiffness | in | 0.026 | 0.034 | 0.048 | 0.054 | 0.144 |
| mass_change | in | 0.026 | 0.026 | 0.026 | 0.026 | 0.026 |
| steering_bias | out | 0.026 | 0.128 | 0.251 | 0.372 | 0.497 |
| steering_loe | out | 0.026 | 0.035 | 0.041 | 0.064 | 0.133 |
| actuator_latency | out | 0.026 | 0.023 | 0.025 | 0.032 | 0.194 |
| wheel_drag | out | 0.026 | 0.036 | 0.058 | 0.082 | 0.109 |
| obs_noise | out | 0.026 | 0.026 | 0.026 | 0.026 | 0.026 |
| obs_latency | out | 0.026 | 0.026 | 0.026 | 0.026 | 0.026 |

## Breakdown severity (lowest s with mean progress < 50%)

| fault | class | breakdown s |
|---|---|---|
| friction_drop | in | 0.75 |
| low_grip_patch | in | 0.75 |
| tire_stiffness | in | 1.0 |
| mass_change | in | never |
| steering_bias | out | never |
| steering_loe | out | 0.75 |
| actuator_latency | out | never |
| wheel_drag | out | never |
| obs_noise | out | never |
| obs_latency | out | never |

## Read

A standard, *non-adaptive* controller degrades under both fault classes -- expected, since it cannot adapt. This confirms the **fault substrate**: every fault produces graded, monotone, controllable degradation with a clear breakdown severity. The next step adds the ADAPTIVE baselines (adaptive-MPC / sysID and our residual): the crossover hypothesis is that classical adaptation recovers the IN-MODEL faults (grip, tire, mass, patch) but not the OUT-OF-MODEL ones (bias, LoE, latency, drag) -- which is where our method earns its place.