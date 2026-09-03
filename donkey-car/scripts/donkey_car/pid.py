"""
pid.py
------
Generic PID controller. No Donkey-specific logic lives here.

P (Proportional) — reacts to the current error magnitude.
I (Integral)     — corrects for long-term drift / steady-state error.
D (Derivative)   — dampens oscillations by reacting to how fast error changes.
"""


class PIDController:

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float | None = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Clamp the accumulated integral to ±integral_limit to prevent
        # windup (the integral ballooning while the car is far off track,
        # then overshooting once it recovers). None disables the clamp.
        self.integral_limit = integral_limit

        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error: float) -> float:
        """Feed in the current error, get back the controller output."""
        self._integral += error
        if self.integral_limit is not None:
            self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        derivative       = error - self._prev_error
        self._prev_error = error

        return (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)

    def reset(self):
        """Clear internal state — call this whenever the car is reset to start."""
        self._integral   = 0.0
        self._prev_error = 0.0
