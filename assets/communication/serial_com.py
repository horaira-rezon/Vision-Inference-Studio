"""
Arduino Communication: connection lifecycle only (connect/disconnect/
is_connected), exactly as before. The nozzle-angle math and step-command
logic that used to live in send_target_angle() here now live in
private/nozzle_targeting.py and are wired in from gui/app.py, which reads
this class's `.connection` attribute to send through.
"""

import time

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class ArduinoLink:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.connection = None

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