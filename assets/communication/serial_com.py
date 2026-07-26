"""
Arduino Communication: same delta/rate-limit stepper logic as your original
script, wrapped so it can be connected/disconnected independently of the
camera and GUI state.
"""

import time

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class ArduinoLink:
    def __init__(self, port, baud, steps_per_degree, command_delay):
        self.port = port
        self.baud = baud
        self.steps_per_degree = steps_per_degree
        self.command_delay = command_delay
        self.connection = None
        self.current_angle = 0.0
        self.last_sent_time = 0.0

    def connect(self):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is not installed")
        self.connection = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(2)  # let the Arduino finish its reset-on-connect

    def disconnect(self):
        if self.connection:
            self.connection.close()
        self.connection = None

    @property
    def is_connected(self):
        return self.connection is not None

    def send_target_angle(self, target_angle_deg):
        angle_delta = target_angle_deg - self.current_angle
        steps = int(round(angle_delta * self.steps_per_degree))
        now = time.time()
        if abs(steps) >= 1 and (now - self.last_sent_time > self.command_delay):
            self.connection.write(f"{(-1 * steps)}\n".encode())
            self.current_angle += steps / self.steps_per_degree
            self.last_sent_time = now