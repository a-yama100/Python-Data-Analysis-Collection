# E:\programming\Project\Portfolio\scripts\potential_customers_script.py

import cv2
import os
import dlib
import numpy as np
import matplotlib.pyplot as plt

# 画像データの読み込み
img1_path = 'img01.jpg'
img2_path = 'img02.jpg'

if os.path.exists(img1_path):
    img1 = cv2.imread(img1_path)
else:
    print(f"Image {img1_path} not found!")

if os.path.exists(img2_path):
    img2 = cv2.imread(img2_path)
else:
    print(f"Image {img2_path} not found!")

# 映像を画像に分割して保存する関数
def save_frames_from_video(video_capture, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        save_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(save_path, frame)
        frame_count += 1

# 映像データの読み込みと分解
mov1_path = 'mov01.avi'
mov2_path = 'mov02.avi'
mov1_frames_dir = 'mov01_frames'
mov2_frames_dir = 'mov02_frames'

if not os.path.exists(mov1_frames_dir) and os.path.exists(mov1_path):
    cap1 = cv2.VideoCapture(mov1_path)
    save_frames_from_video(cap1, mov1_frames_dir)
    cap1.release()

if not os.path.exists(mov2_frames_dir) and os.path.exists(mov2_path):
    cap2 = cv2.VideoCapture(mov2_path)
    save_frames_from_video(cap2, mov2_frames_dir)
    cap2.release()

# 人物を検出する関数
def detect_people(img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(16, 16), scale=1.05)
    return rects

# 顔検出のためのカスケード分類器を読み込む
face_cascade_path = 'haarcascade_frontalface_alt.xml'
if os.path.exists(face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
else:
    raise FileNotFoundError(f"File {face_cascade_path} not found!")

# 顔の向き検出のためのモデルを読み込む
face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
if os.path.exists(face_landmark_path):
    face_landmark_predictor = dlib.shape_predictor(face_landmark_path)
else:
    raise FileNotFoundError(f"File {face_landmark_path} not found!")

# 画像内の人の存在を検出する関数
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def calculate_face_orientation(landmarks):
    # 鼻の先端、目の両端のランドマークを取得
    nose_tip = landmarks.part(30).x
    left_eye = landmarks.part(39).x
    right_eye = landmarks.part(42).x

    # 顔の向きを計算
    if nose_tip < left_eye:
        return "右"
    elif nose_tip > right_eye:
        return "左"
    else:
        return "中"

# 顔の向きを検出する関数
def detect_face_orientation(img, faces):
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        # ここで顔のランドマーク検出や顔の向きの計算を行う
        # (この部分は概略的で、具体的な処理は続く)
        face_landmarks = face_landmark_predictor(face, dlib.rectangle(0, 0, w, h))
        # 顔の向きを計算
        face_orientation = calculate_face_orientation(face_landmarks)
        # 顔の向きを画像に描画
        cv2.putText(img, face_orientation, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 画像から人物を検出
people1 = detect_people(img1)
people2 = detect_people(img2)

def draw_people(img, people_rects):
    for (x, y, w, h) in people_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 人物を画像に描画
draw_people(img1, people1)
draw_people(img2, people2)

# 画像から顔を検出
faces1 = detect_faces(img1)
faces2 = detect_faces(img2)

# 顔の向きを検出
detect_face_orientation(img1, faces1)
detect_face_orientation(img2, faces2)

# 情報の統合と可視化
def create_timelapse(input_dir, output_file):
    img_array = []
    files = sorted(os.listdir(input_dir))
    for filename in files:
        img = cv2.imread(os.path.join(input_dir, filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

# タイムラプス動画から時間帯ごとの人の数を計算
def calculate_people_counts_from_frames(frame_dir):
    files = sorted(os.listdir(frame_dir))
    people_counts = []
    for filename in files:
        img = cv2.imread(os.path.join(frame_dir, filename))
        people_rects = detect_people(img)
        people_counts.append(len(people_rects))
    return people_counts

people_counts_mov1 = calculate_people_counts_from_frames('mov01_frames')
people_counts_mov2 = calculate_people_counts_from_frames('mov02_frames')

# 移動平均を計算
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

ma_people_counts_mov1 = moving_average(people_counts_mov1)
ma_people_counts_mov2 = moving_average(people_counts_mov2)

# グラフの可視化 (時間帯ごとの人の数)
plt.figure(figsize=(12, 6))
plt.plot(people_counts_mov1, label="People Counts (mov1)", color="blue", alpha=0.5)
plt.plot(ma_people_counts_mov1, label="Moving Average (mov1)", color="blue")
plt.plot(people_counts_mov2, label="People Counts (mov2)", color="red", alpha=0.5)
plt.plot(ma_people_counts_mov2, label="Moving Average (mov2)", color="red")
plt.xlabel('Frames')
plt.ylabel('Number of People')
plt.title('Number of People Detected in Each Frame')
plt.legend()
plt.savefig('people_counts_graph.png')  # グラフを保存
plt.show()

# タイムラプス動画を作成
create_timelapse('mov01_frames', 'timelapse1.avi')
create_timelapse('mov02_frames', 'timelapse2.avi')

# グラフの可視化 (顔の数)
face_counts = [len(faces1), len(faces2)]
img_labels = ['img01', 'img02']
plt.bar(img_labels, face_counts)
plt.xlabel('Images')
plt.ylabel('Number of Faces')
plt.title('Number of Faces Detected in Each Image')
plt.savefig('faces_counts_graph.png')  # グラフを保存
plt.show()