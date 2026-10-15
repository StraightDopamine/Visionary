import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download Model Tasks if missing
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(HAND_MODEL_PATH):
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

if not os.path.exists(POSE_MODEL_PATH):
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

# 2. Skeleton connection maps
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Torso
    (11, 13), (13, 15),                    # Left arm
    (12, 14), (14, 16),                    # Right arm
    (23, 25), (25, 27),                    # Left leg
    (24, 26), (26, 28)                     # Right leg
]

# 3. 3D Cube Builder
def create_cube(size=140.0):
    half = size / 2.0
    vertices = np.array([
        [-half, -half, -half], [ half, -half, -half],
        [ half,  half, -half], [-half,  half, -half],
        [-half, -half,  half], [ half, -half,  half],
        [ half,  half,  half], [-half,  half,  half]
    ])
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [1, 2, 6, 5], [0, 3, 7, 4]
    ]
    return vertices, edges, faces

def get_rotation_matrix(rx, ry, rz):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def project_points(points, width, height, focal_length=600.0):
    projected = []
    cx, cy = width / 2.0, height / 2.0
    for p in points:
        z = max(p[2], 1.0)
        x2d = int(cx + (p[0] * focal_length) / z)
        y2d = int(cy + (p[1] * focal_length) / z)
        projected.append((x2d, y2d))
    return projected

def draw_hand_skeleton(image, landmarks, w, h, line_color=(0, 255, 0), point_color=(0, 0, 255)):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(image, pts[p1], pts[p2], line_color, 2)
    for pt in pts:
        cv2.circle(image, pt, 5, point_color, -1)

def draw_body_skeleton(image, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    if len(pts) > 0:
        cv2.circle(image, pts[0], 28, (0, 255, 255), 3)
    for p1, p2 in POSE_CONNECTIONS:
        if p1 < len(pts) and p2 < len(pts):
            cv2.line(image, pts[p1], pts[p2], (0, 255, 0), 3)
    for i, pt in enumerate(pts):
        if i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            cv2.circle(image, pt, 6, (0, 0, 255), -1)

def detect_single_hand_gesture(landmarks):
    wrist = landmarks[0]
    def dist(lm1, lm2): return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
    index_ext  = dist(landmarks[8], wrist)  > dist(landmarks[6], wrist)
    middle_ext = dist(landmarks[12], wrist) > dist(landmarks[10], wrist)
    ring_ext   = dist(landmarks[16], wrist) > dist(landmarks[14], wrist)
    pinky_ext  = dist(landmarks[20], wrist) > dist(landmarks[18], wrist)
    if index_ext and middle_ext and ring_ext and pinky_ext: return "MOVE"
    elif index_ext and not middle_ext and not ring_ext and not pinky_ext: return "ROTATE"
    return "FREEZE"

def is_fist_gesture(landmarks):
    def dist2d(i, j): return np.sqrt((landmarks[i].x - landmarks[j].x)**2 + (landmarks[i].y - landmarks[j].y)**2)
    index_closed = dist2d(8, 0) < dist2d(5, 0)
    middle_closed = dist2d(12, 0) < dist2d(9, 0)
    ring_closed = dist2d(16, 0) < dist2d(13, 0)
    pinky_closed = dist2d(20, 0) < dist2d(17, 0)
    return index_closed and middle_closed and ring_closed and pinky_closed

def get_hand_pitch_angle(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    vec = middle_mcp - wrist
    return np.arctan2(vec[2], vec[1])

def draw_stick_figure(img, x, base_y, state):
    color = (255, 255, 255)
    base_y = int(base_y)
    if state == "CROUCH":
        cv2.circle(img, (x, base_y - 30), 10, color, 2)
        cv2.line(img, (x, base_y - 20), (x, base_y - 10), color, 2)
        cv2.line(img, (x, base_y - 10), (x - 20, base_y), color, 2)
        cv2.line(img, (x, base_y - 10), (x + 20, base_y), color, 2)
        cv2.line(img, (x, base_y - 15), (x + 15, base_y - 5), color, 2)
    else:
        cv2.circle(img, (x, base_y - 60), 10, color, 2)
        cv2.line(img, (x, base_y - 50), (x, base_y - 25), color, 2)
        cv2.line(img, (x, base_y - 25), (x - 15, base_y), color, 2)
        cv2.line(img, (x, base_y - 25), (x + 15, base_y), color, 2)
        cv2.line(img, (x, base_y - 40), (x - 20, base_y - 20), color, 2)
        cv2.line(img, (x, base_y - 40), (x + 20, base_y - 20), color, 2)

def check_collision(px1, py1, px2, py2, ox1, oy1, ox2, oy2):
    return not (px2 < ox1 or px1 > ox2 or py2 < oy1 or py1 > oy2)

def main():
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.75
    )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    WINDOW_NAME = "Gesture Interface"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3D Cube State
    cube_verts, cube_edges, cube_faces = create_cube(size=140.0)
    cube_pos = np.array([0.0, 0.0, 500.0])
    cube_rot = np.array([0.0, 0.0, 0.0])
    prev_single_finger_pos = None
    prev_two_hand_steer_angle = None
    prev_two_hand_pitch_angle = None
    ROTATION_SPEED_MULTIPLIER = 3.5

    # Application Modes
    APP_MODE = "CUBE"
    is_fullscreen = False

    # Game Config & Difficulty
    difficulty = "EASY"
    GAME_STATE = "PROMPT"
    game_over_reason = ""
    score = 0
    start_time = 0
    obstacles = []
    
    # Spawn timers & streak trackers
    last_gh_spawn_time = 0
    last_falling_spawn_time = 0
    gh_consecutive_type = None
    gh_consecutive_count = 0

    # Physics Engine & Anti-Cheat State
    player_y = 500.0
    player_vy = 0.0
    is_grounded = True
    GRAVITY = 0.70
    FAST_FALL_GRAVITY = 2.80
    JUMP_IMPULSE = -12.5
    air_start_time = None
    
    dirt_dots = [[random.randint(0, 1280), random.randint(510, 720)] for _ in range(100)]

    with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            num_detected = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0

            # MODE: 3D CUBE MANIPULATION
            if APP_MODE == "CUBE":
                gesture_text = "FREEZE"
                if num_detected > 0:
                    for landmarks in hand_result.hand_landmarks:
                        draw_hand_skeleton(frame, landmarks, w, h)

                if num_detected == 2:
                    h1, h2 = hand_result.hand_landmarks[0], hand_result.hand_landmarks[1]
                    w1 = np.array([h1[0].x * w, h1[0].y * h])
                    w2 = np.array([h2[0].x * w, h2[0].y * h])
                    curr_steer_angle = np.arctan2(w2[1] - w1[1], w2[0] - w1[0])
                    curr_pitch_angle = (get_hand_pitch_angle(h1) + get_hand_pitch_angle(h2)) / 2.0

                    if prev_two_hand_steer_angle is not None and prev_two_hand_pitch_angle is not None:
                        delta_steer = curr_steer_angle - prev_two_hand_steer_angle
                        if delta_steer > np.pi: delta_steer -= 2 * np.pi
                        elif delta_steer < -np.pi: delta_steer += 2 * np.pi
                        cube_rot[2] += delta_steer * ROTATION_SPEED_MULTIPLIER
                        cube_rot[0] += (curr_pitch_angle - prev_two_hand_pitch_angle) * ROTATION_SPEED_MULTIPLIER

                    prev_two_hand_steer_angle = curr_steer_angle
                    prev_two_hand_pitch_angle = curr_pitch_angle
                    prev_single_finger_pos = None
                    gesture_text = "TWO-HAND ROTATE (FAST)"

                elif num_detected == 1:
                    prev_two_hand_steer_angle, prev_two_hand_pitch_angle = None, None
                    landmarks = hand_result.hand_landmarks[0]
                    gesture_text = detect_single_hand_gesture(landmarks)
                    curr_finger_pos = np.array([landmarks[8].x * w, landmarks[8].y * h])

                    if gesture_text == "MOVE":
                        cube_pos[0], cube_pos[1] = curr_finger_pos[0] - (w / 2.0), curr_finger_pos[1] - (h / 2.0)
                    elif gesture_text == "ROTATE" and prev_single_finger_pos is not None:
                        delta = curr_finger_pos - prev_single_finger_pos
                        cube_rot[1] += delta[0] * 0.01
                        cube_rot[0] += delta[1] * 0.01
                    prev_single_finger_pos = curr_finger_pos
                else:
                    prev_single_finger_pos, prev_two_hand_steer_angle, prev_two_hand_pitch_angle = None, None, None

                R = get_rotation_matrix(cube_rot[0], cube_rot[1], cube_rot[2])
                pts_2d = project_points((cube_verts @ R.T) + cube_pos, w, h)
                overlay = frame.copy()
                
                for face in cube_faces:
                    cv2.fillConvexPoly(overlay, np.array([pts_2d[i] for i in face], np.int32), (180, 180, 180))
                frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
                
                for edge in cube_edges:
                    cv2.line(frame, pts_2d[edge[0]], pts_2d[edge[1]], (255, 255, 255), 2)

                cv2.putText(frame, f"Mode: {gesture_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, "Press 'G' for Game | Press 'T' for Skeleton Tracker", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # MODE: FULL-BODY SKELETON MAPPING TRACKER
            elif APP_MODE == "SKELETON":
                frame = np.zeros_like(frame)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    draw_body_skeleton(frame, pose_result.pose_landmarks[0], w, h)

                if num_detected > 0:
                    for i, landmarks in enumerate(hand_result.hand_landmarks):
                        line_col = (255, 255, 0) if i == 0 else (255, 0, 255)
                        point_col = (0, 255, 255) if i == 0 else (255, 255, 255)
                        draw_hand_skeleton(frame, landmarks, w, h, line_color=line_col, point_color=point_col)

                cv2.putText(frame, "FULL-BODY SKELETON MAPPING TRACKER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'T' to Toggle back to Previous Mode", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # MODE: DINOSAUR GAME
            elif APP_MODE == "GAME":
                cv2.line(frame, (0, 500), (w, 500), (255, 255, 255), 3)
                scroll_speed = 10 if difficulty == "EASY" else 15
                for i in range(len(dirt_dots)):
                    dx, dy = dirt_dots[i]
                    dx -= scroll_speed if GAME_STATE == "PLAYING" else 2
                    if dx < 0: dx = w
                    dirt_dots[i][0] = dx
                    cv2.circle(frame, (dx, dy), 2, (255, 255, 255), -1)

                player_x = 200
                player_state = "STAND"
                hand_lifted = False

                if num_detected == 1:
                    landmarks = hand_result.hand_landmarks[0]
                    if is_fist_gesture(landmarks):
                        player_state = "CROUCH"
                    elif landmarks[0].y < 0.78:
                        hand_lifted = True

                if hand_lifted and is_grounded and player_state != "CROUCH":
                    player_vy = JUMP_IMPULSE
                    is_grounded = False
                    air_start_time = time.time()

                if not is_grounded:
                    player_y += player_vy
                    if player_state == "CROUCH":
                        player_vy += FAST_FALL_GRAVITY
                    else:
                        player_vy += GRAVITY

                    if player_y >= 500.0:
                        player_y = 500.0
                        player_vy = 0.0
                        is_grounded = True
                        air_start_time = None
                    else:
                        if air_start_time is not None and (time.time() - air_start_time > 7.0):
                            GAME_STATE = "GAMEOVER"
                            game_over_reason = "AIR EXHAUSTION! REMAINED IN AIR > 7 SECONDS!"

                if GAME_STATE == "PROMPT":
                    cv2.putText(frame, "GENTLE GESTURE DINO RUNNER", (w//2 - 260, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(frame, f"Goal: Reach 500 Points | Mode: {difficulty} (Press 'D' to toggle)", (w//2 - 280, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, "Controls:", (w//2 - 260, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, "- Slight hand elevation to Jump", (w//2 - 260, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(frame, "- CLOSED FIST to Crouch / Fast-Fall", (w//2 - 260, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(frame, "- WATCH OUT for High Velocity Falling Spikes!", (w//2 - 260, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    cv2.putText(frame, "Show an OPEN HAND to Start", (w//2 - 200, 560), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    
                    if num_detected == 1 and player_state == "STAND":
                        GAME_STATE = "PLAYING"
                        start_time = time.time()
                        obstacles.clear()
                        score = 0
                        player_y = 500.0
                        player_vy = 0.0
                        is_grounded = True
                        air_start_time = None
                        last_gh_spawn_time = time.time()
                        last_falling_spawn_time = time.time()
                        gh_consecutive_type = None
                        gh_consecutive_count = 0

                elif GAME_STATE == "PLAYING":
                    score = int((time.time() - start_time) * 20)
                    cv2.putText(frame, f"Score: {score}/500", (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(frame, f"Difficulty: {difficulty}", (w - 260, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if not is_grounded and air_start_time is not None:
                        air_dur = time.time() - air_start_time
                        cv2.putText(frame, f"Air Time: {air_dur:.1f}s / 7.0s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                    obs_speed = 14 if difficulty == "EASY" else 19

                    # Ground & High Spikes: Enforced anti-streak logic (max 2 consecutive of same type)
                    gh_min, gh_max = (1.5, 2.3) if difficulty == "EASY" else (1.0, 1.6)
                    if time.time() - last_gh_spawn_time > random.uniform(gh_min, gh_max):
                        if gh_consecutive_count >= 2:
                            gh_type = "HIGH_SPIKE" if gh_consecutive_type == "GROUND_SPIKE" else "GROUND_SPIKE"
                        else:
                            gh_type = random.choice(["GROUND_SPIKE", "HIGH_SPIKE"])

                        if gh_type == gh_consecutive_type:
                            gh_consecutive_count += 1
                        else:
                            gh_consecutive_type = gh_type
                            gh_consecutive_count = 1

                        obstacles.append({
                            "type": gh_type,
                            "x": w + 50,
                            "y": 0.0,
                            "vy": 0.0
                        })
                        last_gh_spawn_time = time.time()

                    # Falling Spikes: Slightly reduced fall velocity range (13.0 to 18.0 down from 20.0 to 28.0)
                    fall_min, fall_max = (1.2, 2.2) if difficulty == "EASY" else (0.7, 1.4)
                    if time.time() - last_falling_spawn_time > random.uniform(fall_min, fall_max):
                        spawn_x = random.randint(50, w + 100)
                        obstacles.append({
                            "type": "FALLING_SPIKE",
                            "x": spawn_x,
                            "y": 10.0,
                            "vy": random.uniform(13.0, 18.0)
                        })
                        last_falling_spawn_time = time.time()

                    # Calculate Player Bounding Box
                    if player_state == "CROUCH":
                        px1, py1, px2, py2 = player_x - 15, int(player_y) - 35, player_x + 15, int(player_y)
                    else:
                        px1, py1, px2, py2 = player_x - 15, int(player_y) - 70, player_x + 15, int(player_y)

                    for obs in obstacles[:]:
                        obs["x"] -= obs_speed
                        ox = int(obs["x"])

                        if obs["type"] == "GROUND_SPIKE":
                            cv2.fillConvexPoly(frame, np.array([[ox, 440], [ox-20, 500], [ox, 500]], np.int32), (255, 255, 255))
                            cv2.fillConvexPoly(frame, np.array([[ox, 440], [ox, 500], [ox+20, 500]], np.int32), (180, 180, 180))
                            if check_collision(px1, py1, px2, py2, ox-15, 445, ox+15, 500):
                                GAME_STATE = "GAMEOVER"
                                game_over_reason = "You just got hit by a ground spike!"

                        elif obs["type"] == "HIGH_SPIKE":
                            cv2.fillConvexPoly(frame, np.array([[ox, 435], [ox-20, 370], [ox, 370]], np.int32), (255, 255, 255))
                            cv2.fillConvexPoly(frame, np.array([[ox, 435], [ox, 370], [ox+20, 370]], np.int32), (180, 180, 180))
                            if check_collision(px1, py1, px2, py2, ox-15, 370, ox+15, 435):
                                GAME_STATE = "GAMEOVER"
                                game_over_reason = "You just got hit by a high spike!"

                        elif obs["type"] == "FALLING_SPIKE":
                            obs["y"] += obs["vy"]
                            oy = int(min(obs["y"], 485))
                            
                            cv2.fillConvexPoly(frame, np.array([[ox, oy + 25], [ox - 15, oy - 15], [ox, oy - 15]], np.int32), (0, 0, 255))
                            cv2.fillConvexPoly(frame, np.array([[ox, oy + 25], [ox, oy - 15], [ox + 15, oy - 15]], np.int32), (0, 165, 255))
                            
                            if check_collision(px1, py1, px2, py2, ox - 15, oy - 15, ox + 15, oy + 25):
                                GAME_STATE = "GAMEOVER"
                                game_over_reason = "You got crushed by a falling spike!"

                        if ox < -50 or obs.get("y", 0) > 600:
                            obstacles.remove(obs)

                    if score >= 500:
                        GAME_STATE = "VICTORY"

                elif GAME_STATE == "GAMEOVER":
                    cv2.putText(frame, "GAME OVER", (w//2 - 150, h//2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.putText(frame, game_over_reason, (w//2 - 250, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    cv2.putText(frame, "Show a CLOSED FIST to Restart", (w//2 - 200, h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    if num_detected == 1 and player_state == "STAND":
                        GAME_STATE = "PROMPT"

                elif GAME_STATE == "VICTORY":
                    cv2.putText(frame, "VICTORY!", (w//2 - 140, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.putText(frame, "Score Target Reached (750 Pts)!", (w//2 - 220, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, "Show a CLOSED FIST to Play Again", (w//2 - 220, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    if num_detected == 1 and player_state == "STAND":
                        GAME_STATE = "PROMPT"

                draw_stick_figure(frame, player_x, player_y, player_state)
                cv2.putText(frame, "Press 'G' for Cube | Press 'T' for Skeleton | Press 'D' for Difficulty", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Global Footer
            cv2.putText(frame, "Press 'F' for Fullscreen | 'ESC' to Exit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, frame)
            
            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key in (ord('f'), ord('F')):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            elif key in (ord('g'), ord('G')):
                APP_MODE = "GAME" if APP_MODE == "CUBE" else "CUBE"
            elif key in (ord('t'), ord('T')):
                APP_MODE = "SKELETON" if APP_MODE != "SKELETON" else "GAME"
            elif key in (ord('d'), ord('D')):
                difficulty = "HARD" if difficulty == "EASY" else "EASY"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
