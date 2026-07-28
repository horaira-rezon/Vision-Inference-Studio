"""
PRIVATE - do not publish / .gitignore this folder.

Exact replication of the calculation procedure from your original
private-project.py: diagonal distance, target angle (signed, via X/Z),
delta-based step conversion, rate-limited serial transmission, and running
nozzle position. Not a single formula, constant, or condition has been
changed from that original version - only reorganized into a reusable
class so gui/app.py can call it without containing the math itself.
"""

import time
import numpy as np

# HARDWARE SETUP DETAILS: NEMA 17 @ 400 steps/rev
STEPS_PER_DEGREE = 400.0 / 360.0  # exact fraction, prevents drift

COMMAND_DELAY = 0.1  # seconds to wait between sending motor movements


class NozzleTargeting:
    def __init__(self):
        self.current_nozzle_angle = 0.0
        self.last_sent_time = time.time()

    def compute(self, point_3d):
        """point_3d: (X, Y, Z) in meters, from RealSense deprojection.
        Returns (diagonal_distance, target_angle_deg, direction_text, steps_to_move).
        Pure calculation - does not touch the serial connection or move
        the nozzle's tracked position; call send() for that."""
        X, Y, Z = point_3d

        # Calculate the direct diagonal distance (Hypotenuse)
        diagonal_distance = np.sqrt(X**2 + Y**2 + Z**2)

        # Using X/Z (not abs(X)/Z) naturally preserves the direction sign:
        # positive = Right of center, negative = Left of center, 0 = centered.
        target_angle_deg = np.degrees(np.arctan(X / Z))
        direction_text = "Right" if target_angle_deg > 0 else "Left" if target_angle_deg < 0 else "Center"

        # Relative step distance from the nozzle's LAST known position
        angle_delta = target_angle_deg - self.current_nozzle_angle
        steps_to_move = int(round(angle_delta * STEPS_PER_DEGREE))

        return diagonal_distance, target_angle_deg, direction_text, steps_to_move

    def send(self, arduino_connection, steps_to_move):
        """Rate-limited serial transmission + running position update.
        arduino_connection should be the raw pyserial Serial object, or
        None if Arduino isn't connected (in which case nothing is sent
        and the tracked nozzle position doesn't change)."""
        current_time = time.time()
        if arduino_connection is not None and abs(steps_to_move) >= 1 and \
                (current_time - self.last_sent_time > COMMAND_DELAY):
            # Sends signed integers like "15\n" or "-22\n"
            arduino_connection.write(f"{(-1 * steps_to_move)}\n".encode())

            actual_angle_moved = steps_to_move / STEPS_PER_DEGREE
            self.current_nozzle_angle += actual_angle_moved
            self.last_sent_time = current_time