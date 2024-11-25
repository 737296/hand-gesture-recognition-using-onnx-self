import cv2
from math import degrees

import numpy as np

from demo.hand_landmark import HandLandmark
from demo.palm_detection import PalmDetection
from utils import CvFpsCalc
from utils.utils import rotate_and_crop_rectangle


def main():
    lines_hand = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
    ]
    palm_detection = PalmDetection()
    hand_landmark = HandLandmark()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    cv2.namedWindow('摄像头画面', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('摄像头画面', 1920, 1080)
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    cap_width = 640
    cap_height = 480

    while True:
        fps = cvFpsCalc.get()
        ret, frame = cap.read()
        if not ret:
            print("无法接收画面，结束程序")
            break

        hands = palm_detection(frame)
        rects = []
        cropted_rotated_hands_images = []
        if len(hands) > 0:
            for hand in hands:
                # hand: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
                sqn_rr_size = hand[0]
                rotation = hand[1]
                sqn_rr_center_x = hand[2]
                sqn_rr_center_y = hand[3]
                wh_ratio = 640 / 480
                x1 = (sqn_rr_center_x - sqn_rr_size / 2) * 640
                y1 = (sqn_rr_center_y + sqn_rr_size * wh_ratio / 2) * 480
                x2 = (sqn_rr_center_x + sqn_rr_size / 2) * 640
                y2 = (sqn_rr_center_y - sqn_rr_size * wh_ratio / 2) * 480
                # cv2.rectangle(
                #     frame,  # 参数 1: 图像
                #     (int(x1), int(y1)),  # 参数 2: 左上角点的坐标
                #     (int(x2), int(y2)),  # 参数 3: 右下角点的坐标
                #     (0, 128, 255),  # 参数 4: 矩形颜色（BGR 格式）
                #     2,  # 参数 5: 线条宽度
                #     cv2.LINE_AA  # 参数 6: 线条类型
                # )

                # hand: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
                cx = int(sqn_rr_center_x * cap_width)
                cy = int(sqn_rr_center_y * cap_height)
                xmin = int((sqn_rr_center_x - (sqn_rr_size / 2)) * cap_width)
                xmax = int((sqn_rr_center_x + (sqn_rr_size / 2)) * cap_width)
                ymin = int((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2)) * cap_height)
                ymax = int((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2)) * cap_height)
                xmin = max(0, xmin)
                xmax = min(cap_width, xmax)
                ymin = max(0, ymin)
                ymax = min(cap_height, ymax)
                degree = degrees(rotation)
                # [boxcount, cx, cy, width, height, degree]
                rects.append([cx, cy, (xmax - xmin), (ymax - ymin), degree])
            rects = np.asarray(rects, dtype=np.float32)
            # print("rects:", rects)
            # 获取旋转角度校正为零度的手掌图像
            cropted_rotated_hands_images = rotate_and_crop_rectangle(
                image=frame,
                rects_tmp=rects,
                operation_when_cropping_out_of_range='padding',
            )

        # ============================================================= HandLandmark
        if len(cropted_rotated_hands_images) > 0:
            hand_landmarks, rotated_image_size_leftrights = hand_landmark(
                images=cropted_rotated_hands_images,
                rects=rects,
            )
            # print("hand_landmarks.shape:",hand_landmarks.shape)
            # print("hand_landmarks.len:", len(hand_landmarks))
            # print("hand_landmarks:", hand_landmarks)
            # print("rotated_image_size_leftrights:", rotated_image_size_leftrights)
            for landmark in hand_landmarks:
                # print("landmark:",landmark)
                lines = np.asarray(
                    [
                        np.array([landmark[point] for point in line]).astype(np.int32) for line in lines_hand
                    ]
                )
                cv2.polylines(
                    frame,
                    lines,
                    False,
                    (255, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                for point in landmark:
                    x, y = point
                    cv2.circle(
                        frame,
                        (x, y),
                        2,
                        (0, 128, 255),
                        -1)

        image = draw_info(frame, fps)
        cv2.imshow('摄像头画面', image)
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_info(image, fps):
    cv2.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


if __name__ == '__main__':
    main()
