import pyrealsense2 as rs
import cv2, numpy as np
import serial
import time

arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Adjust COM port as needed
# Wait for the serial connection to initialize before sending commands to prevent buffer overflow and ensure the Arduino is ready to receive data (especially important if the Arduino resets when the serial connection is opened, which is common) - this is CRUCIAL to prevent the first few commands from being lost and to ensure that the nozzle starts in a known position before we begin sending movement commands - without this, the first few commands might be sent before the Arduino is ready, causing it to miss those commands and start in an unknown position, which would throw off all subsequent angle calculations and movements.
time.sleep(2)

# Global variables to track mouse position
mouse_x, mouse_y = 0, 0
is_mouse_initialized = False

# CRITICAL FIX: Track where the nozzle is currently physically pointing to calculate relative movements instead of absolute angles, which prevents the nozzle from trying to "jump" back to center every time and instead only moves the necessary amount to reach the new target angle. This is essential for smooth and accurate tracking of the weeds as the user clicks around the stream. By keeping track of the current nozzle angle, we can calculate how much it needs to move relative to its current position rather than trying to move to an absolute angle every time, which would cause erratic movements and overshooting.
current_nozzle_angle = 0.0

# HARDWARE SETUP DETAILS: NEMA 17 @ 400 steps/rev
STEPS_PER_DEGREE = 400.0 / 360.0  # Exact fraction (1.1111...) to prevent drift

last_sent_time = time.time()
COMMAND_DELAY = 0.1  # Seconds to wait between sending motor movements (adjust based on motor speed)
# 150ms is ideal for NEMA 17 at ~1.5 RPS execution speeds (1.5 RPS * 400 steps/rev = 600 steps/sec)
# Each step takes 1.6 ms (1/600 steps = 0.0016 seconds), so in 150ms the motor can execute about 90 steps.
# This means we can safely send commands that require up to 90 steps at once without overwhelming the buffer.
# If a weed is detected at +30 degrees, we need to send all 33 (30*STEPS_PER_DEGREE) steps at once which would be taking 52.8 ms.
# Then we have to wait for the rest of the milliseconds, so that the nozzle catches up before sending more commands.
# If we need to move from +30 degrees to -30 degrees, that's a 60 degree change, which is 66 steps and 105.6 ms. We can send that all at once, but then we must wait for the nozzle to execute those 66 steps before sending any more commands.

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, is_mouse_initialized
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        is_mouse_initialized = True

# Initialize the RealSense pipeline and configure the streams
pipeline = rs.pipeline()
config = rs.config()

# Enable Depth and Color streams at identical resolutions and frame rates for perfect alignment and to ensure that the intrinsics we get are accurate for our pixel-to-point deprojection calculations. This is CRUCIAL for accurate angle estimation and to prevent distortion in the visual feedback that the user relies on to position the mouse correctly over the weeds in the stream.
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the camera pipeline with the specified configuration and get the active profile to access stream intrinsics. This is CRUCIAL to ensure that we have the correct camera parameters for our depth-to-color alignment and for the pixel-to-point deprojection calculations that are essential for accurate nozzle angle estimation.
profile = pipeline.start(config)

# Get the color camera intrinsics (since depth is aligned to color) to use for deprojection of pixel coordinates to 3D points. This is CRUCIAL for accurate angle estimation, as the intrinsics define how the 2D pixel coordinates map to real-world 3D coordinates, which directly affects the calculation of the target angle for the nozzle.
# Get Intrinsics DIRECTLY from the profile stream BEFORE the loop to ensure we have the correct parameters for the aligned frames, and to prevent any issues with trying to access intrinsics from frames that might not be available yet in the loop. This is CRUCIAL for accurate angle estimation and to prevent errors in the deprojection calculations that rely on these intrinsics.
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()

# Use the camera intrinsics to determine the frame dimensions instead of an undefined video capture object.
width = int(intrinsics.width)
height = int(intrinsics.height)

# Extract standard intrinsic pixel values
fx = intrinsics.fx
fy = intrinsics.fy
cx = intrinsics.ppx
cy = intrinsics.ppy

print("\n" + "="*40)
print("      CAMERA INTRINSICS (PIXELS)      ")
print("="*40)
# The :.0f forces Python to print it with 0 decimal places (like an integer)
print(f"fx : {fx:.0f}")
print(f"fy : {fy:.0f}")
print(f"cx : {cx:.0f}")
print(f"cy : {cy:.0f}")
print("="*40 + "\n")

# Align depth frame to color frame so pixel coordinates match perfectly
align_to = rs.stream.color
align = rs.align(align_to)

# Setup OpenCV window and attach the mouse controller
cv2.namedWindow("Nozzle Angle Estimation", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Nozzle Angle Estimation", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Nozzle Angle Estimation", mouse_callback)

# Flag to handle printing intrinsics exactly once
intrinsics_printed = False

print("Click anywhere on the stream window. Press 'Esc' to quit.")

try:
    while True:
        # Wait for camera frames and align them so that depth and color data correspond to the same pixel coordinates. This is CRUCIAL for accurate angle estimation, as it ensures that the depth value we get for the mouse pointer corresponds to the correct location in the color stream, which is essential for calculating the correct 3D point and thus the correct nozzle angle.
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert color frame to a plottable numpy image array for OpenCV processing and visualization. This is CRUCIAL for drawing the visual feedback (like the axes, lines, and circles) that the user relies on to position the mouse correctly over the weeds in the stream, and to display the distance and angle information directly on the stream for real-time feedback.
        color_image = np.asanyarray(color_frame.get_data())

        # USE ACTUAL INTRINSIC CENTER (cx, cy) AS STREAM CENTER FOR ALL CALCULATIONS AND VISUALIZATIONS TO ENSURE THAT THE ANGLE ESTIMATION IS BASED ON THE TRUE OPTICAL CENTER OF THE CAMERA, WHICH PREVENTS DISTORTION IN THE ANGLE CALCULATIONS AND ENSURES THAT THE NOZZLE MOVES ACCURATELY IN RESPONSE TO THE USER'S MOUSE POSITION RELATIVE TO THE TRUE CENTER OF THE STREAM. This is CRUCIAL for accurate angle estimation and to ensure that the visual feedback is correctly aligned with the user's interactions.
        center_x = int(cx)
        center_y = int(cy)

        # Default mouse to the intrinsic center before the user moves it
        if not is_mouse_initialized:
            mouse_x, mouse_y = center_x, center_y

        # DRAW AXES EXTENDING FROM THE INTRINSIC CENTER (cx, cy)
        # Draw Horizontal Axis Line across the screen width
        cv2.line(color_image, (0, center_y), (intrinsics.width, center_y), (100, 100, 100), 1)
        # Draw Vertical Axis Line down the screen height
        cv2.line(color_image, (center_x, 0), (center_x, intrinsics.height), (100, 100, 100), 1)

        # Get the perpendicular distance (Z in meters) at the mouse pointer
        z_depth = depth_frame.get_distance(mouse_x, mouse_y)

        if z_depth > 0:
            # Project the 2D mouse pixel into real 3D space (X, Y, Z)
            point_3d = rs.rs2_deproject_pixel_to_point(
                intrinsics, [mouse_x, mouse_y], z_depth
            )
            # Correct formula: X**2, Y**2, Z**2 (using python exponent formatting **)
            X, Y, Z = point_3d

            # Calculate the direct diagonal distance (Hypotenuse)
            diagonal_distance = np.sqrt(X**2 + Y**2 + Z**2)

            # --- MATH FIX: CALCULATE SIGNED ABSOLUTE TARGET ANGLE ---
            # Using X/Z instead of abs(X)/Z naturally preserves the direction sign of the anle
            # Positive (+) means Right of center, Negative (-) means Left of center, and 0 means perfectly centered.
            target_angle_deg = np.degrees(np.arctan(X / Z))

            direction_text = "Right" if target_angle_deg > 0 else "Left" if target_angle_deg < 0 else "Center"

            # --- DELTA FIX: CALCULATE RELATIVE STEP DISTANCE ---
            # Calculate how much the nozzle actually needs to rotate from its LAST position
            angle_delta = target_angle_deg - current_nozzle_angle

            # Convert the angular change into hardware stepper steps (1.11 steps per degree)
            steps_to_move = int(round(angle_delta * STEPS_PER_DEGREE))

            # --- SERIAL TRANSMISSION WITH RATE LIMITING ---
            current_time = time.time()
            # Only send a command if the target moved by more than ~1 step,
            # and enough time has passed for the physical nozzle to execute the last move.
            if abs(steps_to_move) >= 1 and (current_time - last_sent_time > COMMAND_DELAY):
                # Only write to serial if the port is actively open and available.
                if arduino:
                    # Sends signed integers like "15\n" or "-22\n"
                    arduino.write(f"{(-1 * steps_to_move)}\n".encode())
                
                # Crucial: Update where the nozzle is physically resting now
                actual_angle_moved = steps_to_move / STEPS_PER_DEGREE
                current_nozzle_angle += actual_angle_moved
                last_sent_time = current_time

                # Terminal Monitoring
                print(f"»» SENT TO ARDUINO: {steps_to_move} steps")
                print(f"Target Ang: {target_angle_deg:.2f}° | Nozzle At: {current_nozzle_angle:.2f}° | Delta Steps: {steps_to_move} ({direction_text})")

            # --- Visualizations ---
            # Draw the direct diagonal line from the intrinsic center to your mouse target
            cv2.line(color_image, (center_x, center_y), (mouse_x, mouse_y), (0, 255, 0), 1)

            # Draw circles at the center (green) and the target (red)
            cv2.circle(color_image, (center_x, center_y), 3, (0, 255, 0), -1)
            cv2.circle(color_image, (mouse_x, mouse_y), 3, (0, 0, 255), -1)

            # SHOW DIAGONAL DISTANCE IN METERS AND ANGLES ON THE STREAM SCREEN
            text_dist = f"Diag Dist: {diagonal_distance:.3f} m"
            text_angle = f"Target Ang: {target_angle_deg:.1f} deg ({direction_text})"
            text_nozzle = f"Nozzle At: {current_nozzle_angle:.1f} deg"

            cv2.putText(color_image, text_dist, (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(color_image, text_angle, (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 2)
            cv2.putText(color_image, text_nozzle, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 2)

        else:
            # Handle out-of-range scenarios gracefully
            cv2.line(color_image, (center_x, center_y), (mouse_x, mouse_y), (0, 0, 255), 1)
            cv2.putText(color_image, "No Depth Data", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the live stream
        cv2.imshow("Nozzle Angle Estimation", color_image)

        # Break loop ONLY when the ESC key (ASCII 27) is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Safely close hardware resources
    pipeline.stop()
    if arduino:
        arduino.close()
    cv2.destroyAllWindows()
    print("Pipeline and serial connections closed safely.")