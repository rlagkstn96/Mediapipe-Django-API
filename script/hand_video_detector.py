import math
import os

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# --- user defined constants ---
LEFT_SHOULDER_VAL = 11
RIGHT_SHOULDER_VAL = 12
LEFT_ELBOW_VAL = 13
RIGHT_ELBOW_VAL = 14
LEFT_WRIST_VAL = 15
RIGHT_WRIST_VAL = 16

LEFT_HIP_VAL = 23
RIGHT_HIP_VAL = 24
LEFT_KNEE_VAL = 25
RIGHT_KNEE_VAL = 26
LEFT_ANKLE_VAL = 27
RIGHT_ANKLE_VAL = 28

BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
# --- user defined constants ---

# this method is used to deduce the hand_video method
# plz refer to the hand_video method accordingly
def hand_image():
    # For static images:
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

    # feed a video:
    videoFile = "test_vid.mp4"
    cap = cv2.VideoCapture(videoFile)
    flag, frame = cap.read()

    # while cap.isOpened():
    while flag:
        image = cv2.flip(frame, 1)
        frame_ID = cap.get(1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_hight, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
            )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(
            '/tmp/annotated_image_' + str(frame_ID) + '.png', cv2.flip(annotated_image, 1))
        flag, frame = cap.read()
    hands.close()

def hand_video(flag, frame):
    # For static images:
    # parameters for the detector
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)
    # flip it along y axis
    image = cv2.flip(frame, 1)
    # color format conversion
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        hands.close()
        return frame
    image_hight, image_width, _ = image.shape
    annotated_image = image.copy()
    # draw result landmarks
    for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        )
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # flip it back and return
    # return cv2.flip(annotated_image, 1)

# save the video if user chooese so
def vid_save():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # return cv2.flip(frame, 1)

def pose_video():
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)
    video = cv2.VideoCapture('mov_data/IMG_9668.MOV')
    # init variables
    walk_count = 0
    frame_no = 0
    left_ankle_forward = False
    right_ankle_forward = False
    walk_sequences = {}
    walk_count_list = []

    while video.isOpened():

        ok, frame = video.read()

        if not ok:
            continue

        # # Flip the frame horizontally for natural (selfie-view) visualization.
        # frame = cv2.flip(frame, 1)

        frame_no = frame_no + 1
        output_frame = frame.copy()
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frameRGB.flags.writeable = False  # to improve performance

        # TODO : 디자인 처리된 이미지로 대체될 예정
        output_frame = display_monitoring_box(output_frame)

        # TODO : 실제 landmark data와 연동 예정
        output_frame = display_body_status(output_frame)

        # ---
        detection_results = pose.process(frameRGB)

        frame_height, frame_width, _ = frame.shape

        if detection_results.pose_landmarks is None:
            continue

        # for testing
        # display_all_body_angle(detection_results, frame)

        lm = detection_results.pose_landmarks.landmark
        plm = mp_pose.PoseLandmark

        left_hip = get_body_coord(plm.LEFT_HIP, frame_height, frame_width, lm, plm)
        left_knee = get_body_coord(plm.LEFT_KNEE, frame_height, frame_width, lm, plm)
        left_ankle = get_body_coord(plm.LEFT_ANKLE, frame_height, frame_width, lm, plm)

        right_hip = get_body_coord(plm.RIGHT_HIP, frame_height, frame_width, lm, plm)
        right_knee = get_body_coord(plm.RIGHT_KNEE, frame_height, frame_width, lm, plm)
        right_ankle = get_body_coord(plm.RIGHT_ANKLE, frame_height, frame_width, lm, plm)

        right_heel = get_body_coord(plm.RIGHT_HEEL, frame_height, frame_width, lm, plm)
        right_foot_index = get_body_coord(plm.RIGHT_FOOT_INDEX, frame_height, frame_width, lm, plm)

        # for testing
        draw_interested_point(left_ankle, left_hip, left_knee, output_frame, right_ankle, right_hip, right_knee)

        # 2개 ankle이 cross over 하면 1 walk로 간주
        left_ankle_x = left_ankle[0]
        right_ankle_x = right_ankle[0]
        foot_size = int(get_distance(right_foot_index, right_heel))

        # count walk : 왼쪽 방향으로 걸을 떄 기준으로 count
        if abs(left_ankle_x - right_ankle_x) > foot_size:
            if not left_ankle_forward and (left_ankle_x < right_ankle_x):
                walk_count = walk_count + 1
                left_ankle_forward = True
                right_ankle_forward = False
                print(f'left walk = {walk_count}, {abs(left_ankle_x - right_ankle_x)}, {foot_size}, {frame_no}')

            if not right_ankle_forward and (left_ankle_x > right_ankle_x):
                walk_count = walk_count + 1
                left_ankle_forward = False
                right_ankle_forward = True
                print(f'right walk = {walk_count}, {abs(left_ankle_x - right_ankle_x)}, {foot_size}, {frame_no}')

        if walk_count == 0:
            continue

        # walk_sequences에 4회 보행 데이터가 쌓이면 평균 각도 계산하여 보행 feedback
        walk_count_list = check_mission(detection_results, frame, walk_count, walk_count_list, walk_sequences)

        draw_landmarks(detection_results, mp_drawing, mp_pose, output_frame)

        cv2.imshow('Twentydot EZMO ', output_frame)
        # cv2.waitKey()

        # stop this program when ESC pressed
        k = cv2.waitKey(1) & 0xFF  # Retreive the ASCII code of the key pressed
        if k == 27:  # ESC
            break

        print(f'last frame no = {frame_no}')

        # Release the VideoCapture object and close the windows.
        video.release()
        cv2.destroyAllWindows()

        cv2.flip(output_frame, 1)

def display_monitoring_box(output_frame):
    # background 이미지
    small_img = cv2.imread("images/background_small.jpg")
    small_img = cv2.resize(small_img, dsize=(300, 600), interpolation=cv2.INTER_AREA)
    output_frame = overlay_image(small_img, output_frame, 10, 50)
    # 팔 이미지
    s_img_arm = cv2.imread("images/arm.png", -1)
    s_img_arm = cv2.resize(s_img_arm, dsize=(80, 80), interpolation=cv2.INTER_AREA)
    output_frame = overlay_transparent_image(s_img_arm, output_frame, 20, 60)
    # 다리 이미지
    s_img_leg = cv2.imread("images/leg.png", -1)
    s_img_leg = cv2.resize(s_img_leg, dsize=(80, 80), interpolation=cv2.INTER_AREA)
    output_frame = overlay_transparent_image(s_img_leg, output_frame, 20, 250)
    # 무릎
    s_img_knee = cv2.imread("images/knee.png", -1)
    s_img_knee = cv2.resize(s_img_knee, dsize=(80, 80), interpolation=cv2.INTER_AREA)
    output_frame = overlay_transparent_image(s_img_knee, output_frame, 20, 450)
    return output_frame


def display_body_status(output_frame):
    # TODO : 실제 landmark data와 연동

    # 팔 상태
    output_frame = display_normal(output_frame, (140, 100), 35)  # 왼쪽
    output_frame = display_normal(output_frame, (230, 100), 35)  # 오른쪽
    # 다리 상태
    output_frame = display_abnormal(output_frame, (140, 290), 35, 133, RED, YELLOW)  # 왼쪽
    output_frame = display_normal(output_frame, (230, 290), 35)  # 오른쪽
    # 무릎 상태
    output_frame = display_normal(output_frame, (140, 480), 35)  # 왼쪽
    output_frame = display_abnormal(output_frame, (230, 480), 35, 173, RED, YELLOW)  # 오른쪽
    return output_frame


def display_normal(img, xy_coord, radius):
    cv2.circle(img, xy_coord, radius, BLUE, -1)

    return img


def overlay_transparent_image(small_img, large_img, x_offset, y_offset):

    y1, y2 = y_offset, y_offset + small_img.shape[0]
    x1, x2 = x_offset, x_offset + small_img.shape[1]

    alpha_s = small_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        large_img[y1:y2, x1:x2, c] = (alpha_s * small_img[:, :, c] +
                                  alpha_l * large_img[y1:y2, x1:x2, c])

    return large_img


def overlay_image(small_img, large_img, x_offset, y_offset):

    large_img[y_offset:y_offset + small_img.shape[0], x_offset:x_offset + small_img.shape[1]] = small_img

    return large_img


def display_abnormal(img, center, radius, angle_degrees, circle_color, hand_color):
    # Convert the angle to radians
    angle_radians = math.radians(angle_degrees)
    # Calculate the start and end points of the line
    start_point = center
    end_point = (
        int(center[0] + radius * math.sin(angle_radians)),
        int(center[1] - radius * math.cos(angle_radians))
    )
    cv2.circle(img, center, radius, circle_color, -1)  # red
    cv2.line(img, center, end_point, hand_color, 2)  # yellow

    return img


def play_sound(audio_folder, audio_file, angle_info):
    audio_path_file = os.path.join(audio_folder, audio_file)
    p = vlc.MediaPlayer(audio_path_file)
    p.play()
    print(f'{angle_info[0]} = {angle_info[1]}, sound_file = {audio_path_file}')


def get_distance(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


def calculate_angle(landmark1, landmark2, landmark3):

    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # adjust if the angle is less than zero.
    if angle < 0:
        angle += 360

    return angle


def get_max_angle(body_1, body_2, body_3, detection_results_list, width, height):
    body_angle_max = -1000

    for detection_results in detection_results_list:

        body_angle = get_angle(body_1, body_2, body_3, detection_results, width, height)
        # print(body_1, body_2, body_3, body_angle)
        if body_angle > body_angle_max:
            body_angle_max = body_angle

    return body_angle_max


def get_angle(body_1, body2, body3, det_results, width, height):

    landmarks = []

    for landmark in det_results.pose_landmarks.landmark:
        landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

    angle = calculate_angle(landmarks[body_1], landmarks[body2], landmarks[body3])
    if angle > 180:
        angle = 360 - angle

    return angle


def display_all_body_angle(detection_results, image):
    height, width, _ = image.shape

    left_leg_angle = get_angle(LEFT_SHOULDER_VAL, LEFT_HIP_VAL, LEFT_KNEE_VAL, detection_results, width, height)
    right_leg_angle = get_angle(RIGHT_SHOULDER_VAL, RIGHT_HIP_VAL, RIGHT_KNEE_VAL, detection_results, width, height)
    left_knee_angle = get_angle(LEFT_HIP_VAL, LEFT_KNEE_VAL, LEFT_ANKLE_VAL, detection_results, width, height)
    right_knee_angle = get_angle(RIGHT_HIP_VAL, RIGHT_KNEE_VAL, RIGHT_ANKLE_VAL, detection_results, width, height)
    left_frontarm_angle = get_angle(LEFT_ELBOW_VAL, LEFT_SHOULDER_VAL, LEFT_HIP_VAL, detection_results, width, height)
    right_frontarm_angle = get_angle(RIGHT_ELBOW_VAL, RIGHT_SHOULDER_VAL, RIGHT_HIP_VAL, detection_results, width, height)

    print(f'left_leg_angle = {left_leg_angle}')
    print(f'right_leg_angle = {right_leg_angle}')
    print(f'left_knee_angle = {left_knee_angle}')
    print(f'right_knee_angle = {right_knee_angle}')
    print(f'left_frontarm_angle = {left_frontarm_angle}')
    print(f'right_frontarm_angle = {right_frontarm_angle}')


# 각 보행의 신체별 평균각도 계산
def get_avg_angles_4walks(walk_sequences, image):
    height, width, _ = image.shape

    left_leg_max_list = []
    right_leg_max_list = []
    left_knee_max_list = []
    right_knee_max_list = []
    left_frontarm_max_list = []
    right_frontarm_max_list = []

    # left_backarm_max_list = []
    # right_backarm_max_list = []

    for _, detection_results_list in walk_sequences.items():

        left_leg_angle_max = get_max_angle(LEFT_SHOULDER_VAL, LEFT_HIP_VAL, LEFT_KNEE_VAL, detection_results_list, width, height)
        right_leg_angle_max = get_max_angle(RIGHT_SHOULDER_VAL, RIGHT_HIP_VAL, RIGHT_KNEE_VAL, detection_results_list, width, height)

        left_knee_angle_max = get_max_angle(LEFT_HIP_VAL, LEFT_KNEE_VAL, LEFT_ANKLE_VAL, detection_results_list, width, height)
        right_knee_angle_max = get_max_angle(RIGHT_HIP_VAL, RIGHT_KNEE_VAL, RIGHT_ANKLE_VAL, detection_results_list, width, height)

        left_frontarm_angle_max = get_max_angle(LEFT_ELBOW_VAL, LEFT_SHOULDER_VAL, LEFT_HIP_VAL, detection_results_list, width, height)
        right_frontarm_angle_max = get_max_angle(RIGHT_ELBOW_VAL, RIGHT_SHOULDER_VAL, RIGHT_HIP_VAL, detection_results_list, width, height)

        # left_backtarm_angle_max = get_max_angle(LEFT_ELBOW_VAL, LEFT_SHOULDER_VAL, LEFT_HIP_VAL, detection_results_list, width, height)
        # right_backtarm_angle_max = get_max_angle(RIGHT_ELBOW_VAL, RIGHT_SHOULDER_VAL, RIGHT_HIP_VAL, detection_results_list, width, height)

        left_leg_max_list.append(left_leg_angle_max)
        right_leg_max_list.append(right_leg_angle_max)
        left_knee_max_list.append(left_knee_angle_max)
        right_knee_max_list.append(right_knee_angle_max)
        left_frontarm_max_list.append(left_frontarm_angle_max)
        right_frontarm_max_list.append(right_frontarm_angle_max)

    left_leg_angle_avg = np.mean(left_leg_max_list)
    right_leg_angle_avg = np.mean(right_leg_max_list)
    left_knee_angle_avg = np.mean(left_knee_max_list)
    right_knee_angle_avg = np.mean(right_knee_max_list)
    left_frontarm_angle_avg = np.mean(left_frontarm_max_list)
    right_frontarm_angle_avg = np.mean(right_frontarm_max_list)

    avg_angles = {
                    'left_leg_angle': left_leg_angle_avg,
                    'right_leg_angle': right_leg_angle_avg,
                    'left_knee_angle': left_knee_angle_avg,
                    'right_knee_angle': right_knee_angle_avg,
                    'left_frontarm_angle': left_frontarm_angle_avg,
                    'right_frontarm_angle': right_frontarm_angle_avg,
                 }
    return avg_angles


def check_leg_feedback(audio_folder, body_angle, audio_file_over, audio_file_below):
    if body_angle > 80:
        angle_info = ('leg_angle', body_angle)
        play_sound(audio_folder, audio_file_over, angle_info)
    elif body_angle < 70:
        angle_info = ('leg_angle', body_angle)
        play_sound(audio_folder, audio_file_below, angle_info)


def check_knee_feedback(audio_folder, body_angle, audio_file_over):
    if body_angle > 15:
        angle_info = ('leg_angle', body_angle)
        play_sound(audio_folder, audio_file_over, angle_info)


def check_frontarm_feedback(audio_folder, body_angle, audio_file_over, audio_file_below):
    if body_angle > 90:
        angle_info = ('frontarm_angle', body_angle)
        play_sound(audio_folder, audio_file_over, angle_info)
    elif body_angle < 80:
        angle_info = ('frontarm_angle', body_angle)
        play_sound(audio_folder, audio_file_below, angle_info)


def feedback_walk_pose(avg_angles_4walks):
    audio_folder = './음성/좌우구분/wav'

    check_leg_feedback(audio_folder, avg_angles_4walks['left_leg_angle'], '좌측다리각도초과.wav', '좌측다리각도미달.wav')
    check_leg_feedback(audio_folder, avg_angles_4walks['right_leg_angle'], '우측다리각도초과.wav', '우측다리각도미달.wav')
    check_knee_feedback(audio_folder, avg_angles_4walks['left_knee_angle'], '좌측무릎각도초과.wav')
    check_knee_feedback(audio_folder, avg_angles_4walks['right_knee_angle'], '우측무릎각도초과.wav')
    check_frontarm_feedback(audio_folder, avg_angles_4walks['left_frontarm_angle'], '좌측팔각도(앞)초과.wav', '좌측팔각도(앞)미달.wav')
    check_frontarm_feedback(audio_folder, avg_angles_4walks['right_frontarm_angle'], '우측팔각도(앞)초과.wav', '우측팔각도(앞)미달.wav')


def check_mission(detection_results, frame, walk_count, walk_count_list, walk_sequences):

    walk_count_list.append(walk_count)
    walks_count_set = set(walk_count_list)
    if not len(walks_count_set) <= 4:
        avg_angles_4walks = get_avg_angles_4walks(walk_sequences, frame)
        feedback_walk_pose(avg_angles_4walks)

        # initialize
        walk_count_list = []
        walk_sequences = {}
        walk_count_list.append(walk_count)

    if walk_sequences.get(walk_count):
        walk_sequences[walk_count] = walk_sequences[walk_count] + [detection_results]
    else:
        walk_sequences[walk_count] = [detection_results]

    # print(f'---> walk count = {walk_count} appended into sequences')

    return walk_count_list


def draw_landmarks(detection_results, mp_drawing, mp_pose, output_frame):
    # draw landmarks
    if detection_results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_frame, landmark_list=detection_results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)


def get_body_coord(body, frame_height, frame_width, lm, plm):
    return int(lm[body].x * frame_width), int(lm[body].y * frame_height)


def draw_interested_point(left_ankle, left_hip, left_knee, output_frame, right_ankle, right_hip, right_knee):
    cv2.circle(output_frame, left_hip, 5, (0, 255, 255), 3)
    cv2.circle(output_frame, left_knee, 5, (0, 255, 255), 3)
    cv2.circle(output_frame, left_ankle, 5, (0, 255, 255), 3)
    cv2.circle(output_frame, right_hip, 5, (255, 255, 0), 3)
    cv2.circle(output_frame, right_knee, 5, (255, 255, 0), 3)
    cv2.circle(output_frame, right_ankle, 5, (255, 255, 0), 3)
