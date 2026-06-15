#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AveFace 纯算法测试（无GUI）
适配 GitHub Codespaces 无头Linux + Xvfb虚拟显示
功能：人脸关键点检测、人脸对齐、平均脸生成、面颊LAB值导出
"""
import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# ===================== 全局环境变量（禁用图形/GPU/EGL，适配无头环境） =====================
os.environ['NO_AT_BRIDGE'] = '1'
os.environ['OS_ACTIVITY_DT_MODE'] = 'disable'
os.environ['TK_SILENCE_DEPRECATION'] = '1'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_alsologtostderr'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['MEDIAPIPE_IGNORE_CLEARCUT'] = '1'
os.environ['MP_NO_TELEMETRY'] = '1'
os.environ['MEDIAPIPE_LOG_LEVEL'] = '3'
os.environ["MEDIAPIPE_DISABLE_EGL"] = "1"
os.environ["MEDIAPIPE_GPU_USE_EGL"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 强制matplotlib使用无头后端
import matplotlib
matplotlib.use('Agg')

# ===================== 依赖导入 =====================
import logging
import traceback
from scipy.spatial import Delaunay
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# 关闭冗余日志
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ("mediapipe", "tensorflow", "absl"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# ===================== 全局关键点索引常量 =====================
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
LEFT_NOSE_WING = 98
RIGHT_NOSE_WING = 327

STABLE_LANDMARK_INDICES = [1, 33, 263, 152, 61, 291]

# ===================== 工具函数：面颊ROI & LAB提取 =====================
def calc_cheek_roi(lm: np.ndarray, img_w: int, img_h: int, side: str):
    try:
        dist_left_eye = abs(lm[LEFT_EYE_OUTER][0] - lm[LEFT_EYE_INNER][0])
        dist_right_eye = abs(lm[RIGHT_EYE_OUTER][0] - lm[RIGHT_EYE_INNER][0])
        square_size = int((dist_left_eye + dist_right_eye) / 2.0)
        if square_size < 5:
            return None

        y_bottom = int((lm[LEFT_NOSE_WING][1] + lm[RIGHT_NOSE_WING][1]) / 2.0)
        y_top = y_bottom - square_size

        if side == "left":
            x_right = int(lm[LEFT_NOSE_WING][0])
            x_left = x_right - square_size
        else:
            x_left = int(lm[RIGHT_NOSE_WING][0])
            x_right = x_left + square_size

        x1 = max(0, x_left)
        y1 = max(0, y_top)
        x2 = min(img_w - 1, x_right)
        y2 = min(img_h - 1, y_bottom)

        if x2 - 10 < 10 or y2 - y1 < 10:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None

def extract_cheek_lab(img: np.ndarray, lm: np.ndarray, side: str):
    h, w = img.shape[:2]
    roi = calc_cheek_roi(lm, w, h, side)
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    cheek_region = img[y1:y2, x1:x2]
    lab = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2LAB)
    lab_mean = lab.reshape(-1, 3).mean(axis=0)
    L = lab_mean[0] * 100.0 / 255.0
    a = lab_mean[1] - 128.0
    b = lab_mean[2] - 128.0
    return np.array([L, a, b], dtype=np.float32)

def get_bilateral_cheek_lab(img: np.ndarray, lm: np.ndarray):
    left_lab = extract_cheek_lab(img, lm, "left")
    right_lab = extract_cheek_lab(img, lm, "right")
    if left_lab is None and right_lab is None:
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)
    if left_lab is None:
        return right_lab
    if right_lab is None:
        return left_lab
    return (left_lab + right_lab) / 2.0

# ===================== 人脸关键点检测器 =====================
class FaceLandmarkDetector:
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._landmarker: Optional[FaceLandmarker] = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"模型不存在: {self._model_path}")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        print("✅ 人脸关键点模型加载成功")

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self._landmarker is None:
            return None
        h, w = image.shape[:2]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            return None

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None
        landmarks = np.array(
            [[p.x * w, p.y * h] for p in result.face_landmarks[0]],
            dtype=np.float32
        )
        if len(landmarks) < 400:
            return None
        return landmarks

    def close(self):
        if self._landmarker:
            self._landmarker.close()

# ===================== 人脸对齐工具类 =====================
class FaceAligner:
    @staticmethod
    def procrustes_align(src_points: np.ndarray, dst_points: np.ndarray):
        src_mean = np.mean(src_points, axis=0)
        dst_mean = np.mean(dst_points, axis=0)
        src_centered = src_points - src_mean
        dst_centered = dst_points - dst_mean

        src_scale = np.sqrt(np.sum(src_centered**2))
        dst_scale = np.sqrt(np.sum(dst_centered**2))
        if src_scale < 1e-10:
            return src_points, None, 1.0

        scale = dst_scale / src_scale
        src_norm = src_centered / src_scale
        dst_norm = dst_centered / dst_scale

        H = src_norm.T @ dst_norm
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = scale * R
        M[:, 2] = dst_mean - scale * (R @ src_mean)
        src_homo = np.hstack([src_points, np.ones((len(src_points), 1))])
        aligned = (M @ src_homo.T).T
        return aligned.astype(np.float32), M, scale

    @staticmethod
    def rigid_align_landmarks(landmarks_list):
        n = len(landmarks_list)
        if n == 0:
            raise ValueError("关键点列表为空")
        init_avg = np.mean(landmarks_list, axis=0)
        aligned_landmarks = []
        transforms = []
        for lm in landmarks_list:
            aligned, M, _ = FaceAligner.procrustes_align(lm, init_avg)
            aligned_landmarks.append(aligned)
            transforms.append(M)
        avg_landmark = np.mean(aligned_landmarks, axis=0)
        return aligned_landmarks, avg_landmark, transforms

    @staticmethod
    def apply_rigid_transform(image: np.ndarray, M):
        if M is None:
            return image
        h, w = image.shape[:2]
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    @staticmethod
    def precise_warp(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray):
        h, w = image.shape[:2]
        boundary = np.array([
            [0, 0], [w // 2, 0], [w - 1, 0],
            [w - 1, h // 3], [w - 1, 2 * h // 3], [w - 1, h - 1],
            [w // 2, h - 1], [0, h - 1],
            [0, 2 * h // 3], [0, h // 3],
        ], dtype=np.float32)
        src_all = np.vstack([src_points, boundary])
        dst_all = np.vstack([dst_points, boundary])
        try:
            tri = Delaunay(src_all)
            simplices = tri.simplices
        except Exception as e:
            print(f"⚠️ Delaunay 三角剖分失败: {e}")
            traceback.print_exc()
            # 回退：使用全局仿射矩阵进行整体变形
            src_stable = src_points[STABLE_LANDMARK_INDICES].astype(np.float32)
            dst_stable = dst_points[STABLE_LANDMARK_INDICES].astype(np.float32)
            M_global, _ = cv2.estimateAffinePartial2D(src_stable, dst_stable)
            if M_global is not None:
                return cv2.warpAffine(
                    image, M_global, (w, h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
            else:
                return image

        accumulator = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for si, simplex in enumerate(simplices):
            try:
                src_tri = src_all[simplex].astype(np.float32)
                dst_tri = dst_all[simplex]
                if cv2.contourArea(src_tri) < 1.0:
                    continue
                M = cv2.getAffineTransform(src_tri, dst_tri)
                if M is None:
                    continue

                x_min_f, y_min_f = dst_tri.min(axis=0)
                x_max_f, y_max_f = dst_tri.max(axis=0)
                x_min = max(0, int(np.floor(x_min_f)))
                x_max = min(w, int(np.ceil(x_max_f)) + 1)
                y_min = max(0, int(np.floor(y_min_f)))
                y_max = min(h, int(np.ceil(y_max_f)) + 1)
                roi_w = x_max - x_min
                roi_h = y_max - y_min
                if roi_w <= 0 or roi_h <= 0:
                    continue

                tri_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                local_tri = dst_tri.copy()
                local_tri[:, 0] -= x_min
                local_tri[:, 1] -= y_min
                cv2.fillConvexPoly(tri_mask, local_tri.astype(np.int32), 255)
                if cv2.countNonZero(tri_mask) == 0:
                    continue

                M_local = M.copy()
                M_local[0, 2] -= x_min
                M_local[1, 2] -= y_min
                roi_buffer = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
                cv2.warpAffine(
                    image, M_local, (roi_w, roi_h), dst=roi_buffer,
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                ys, xs = np.nonzero(tri_mask > 0)
                dst_ys = y_min + ys
                dst_xs = x_min + xs

                # 逐像素赋值，修复维度不匹配报错
                for y, x, dy, dx in zip(ys, xs, dst_ys, dst_xs):
                    accumulator[dy, dx] += roi_buffer[y, x]
                    weight_map[dy, dx] += 1.0
            except Exception as e:
                print(f"⚠️ 处理三角形索引 {si} 时发生异常: {e}")
                traceback.print_exc()
                continue

        valid_mask = weight_map > 0
        warped = np.zeros((h, w, 3), dtype=np.float32)
        valid_y, valid_x = np.nonzero(valid_mask)
        if len(valid_y) > 0:
            weights = weight_map[valid_y, valid_x][:, np.newaxis]
            warped[valid_y, valid_x] = accumulator[valid_y, valid_x] / weights

        unfilled = ~valid_mask
        if np.any(unfilled):
            src_stable = src_points[STABLE_LANDMARK_INDICES].astype(np.float32)
            dst_stable = dst_points[STABLE_LANDMARK_INDICES].astype(np.float32)
            M_global, _ = cv2.estimateAffinePartial2D(src_stable, dst_stable)
            if M_global is not None:
                global_warped = cv2.warpAffine(
                    image, M_global, (w, h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                warped[unfilled] = global_warped[unfilled].astype(np.float32)
        return np.clip(warped, 0, 255).astype(np.uint8)

# ===================== 核心业务函数 =====================
def run_average_face(img_dir: str, model_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    if len(img_paths) < 2:
        print(f"❌ 图片数量不足，至少需要2张，当前：{len(img_paths)}")
        return

    print(f"📸 读取图片: {len(img_paths)} 张")
    detector = FaceLandmarkDetector(model_path)
    all_imgs = []
    all_lms = []
    valid_paths = []

    MAX_DIM = 1024
    for idx, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ 无法读取: {os.path.basename(path)}")
            continue
        h0, w0 = img.shape[:2]
        max_side = max(h0, w0)
        if max_side > MAX_DIM:
            scale = MAX_DIM / float(max_side)
            new_w = int(w0 * scale)
            new_h = int(h0 * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  · 缩放图片 {os.path.basename(path)} -> {new_w}x{new_h}")
        lm = detector.detect(img)
        if lm is None:
            print(f"⚠️ 未检测到人脸: {os.path.basename(path)}")
            continue
        all_imgs.append(img)
        all_lms.append(lm)
        valid_paths.append(path)
    detector.close()

    # 统一所有图片尺寸（右/下填充），避免后续堆叠时形状不一致
    if len(all_imgs) > 0:
        max_h = max(img.shape[0] for img in all_imgs)
        max_w = max(img.shape[1] for img in all_imgs)
        for i, img in enumerate(all_imgs):
            h, w = img.shape[:2]
            if h != max_h or w != max_w:
                pad_bottom = max_h - h
                pad_right = max_w - w
                img_padded = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_REFLECT)
                all_imgs[i] = img_padded

    if len(all_lms) < 2:
        print("❌ 有效人脸不足，终止运行")
        return

    print("🔧 执行人脸刚性对齐...")
    aligned_lms, avg_lm, transforms = FaceAligner.rigid_align_landmarks(all_lms)

    print("🔧 执行精细Delaunay三角对齐...")
    aligned_imgs = []
    for img, lm_aligned, M in zip(all_imgs, aligned_lms, transforms):
        try:
            idx = len(aligned_imgs) + 1
            print(f"  - 对齐图像 {idx}/{len(all_lms)} ...")
            img_rigid = FaceAligner.apply_rigid_transform(img, M)
            warped = FaceAligner.precise_warp(img_rigid, lm_aligned, avg_lm)
            aligned_imgs.append(warped)
            print(f"    -> 完成 图像 {idx}")
        except Exception as e:
            print(f"⚠️ 对齐过程中发生异常: {e}")
            traceback.print_exc()
            # 使用刚性变换后的图像作为后备
            aligned_imgs.append(img_rigid)

    print("🧮 计算像素均值，生成平均脸...")
    stack = np.array(aligned_imgs, dtype=np.float64)
    avg_face = np.mean(stack, axis=0)
    avg_face = np.clip(avg_face, 0, 255).astype(np.uint8)

    # 保存平均脸
    avg_save_path = os.path.join(out_dir, "1_Average_Face.png")
    cv2.imwrite(avg_save_path, avg_face)
    print(f"✅ 平均脸已保存: {avg_save_path}")

    # 保存平均关键点图
    h, w = avg_face.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for (x, y) in avg_lm:
        cv2.circle(canvas, (int(x), int(y)), 2, (255, 0, 0), -1)
    lm_save_path = os.path.join(out_dir, "2_Average_Landmarks.png")
    cv2.imwrite(lm_save_path, canvas)
    print(f"✅ 平均关键点图已保存: {lm_save_path}")

    # 导出面颊LAB数据CSV
    lab_csv = os.path.join(out_dir, "Cheek_LAB.csv")
    with open(lab_csv, "w", encoding="utf-8-sig") as f:
        f.write("FileName,L_L,A_L,B_L,L_R,A_R,B_R\n")
        for img, lm, path in zip(all_imgs, all_lms, valid_paths):
            name = os.path.basename(path)
            if lm is None:
                f.write(f"{name},N/A,N/A,N/A,N/A,N/A,N/A\n")
                continue
            l_lab = extract_cheek_lab(img, lm, "left")
            r_lab = extract_cheek_lab(img, lm, "right")

            def fmt(arr):
                return f"{arr[0]:.2f},{arr[1]:.2f},{arr[2]:.2f}" if arr is not None else "N/A,N/A,N/A"

            f.write(f"{name},{fmt(l_lab)},{fmt(r_lab)}\n")
    print(f"✅ 面颊LAB数据已保存: {lab_csv}")

    print("\n🎉 全部任务执行完毕！")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 配置路径（与仓库文件保持一致）
    MODEL_FILE = "face_landmarker.task"
    IMAGE_FOLDER = "test_imgs"
    OUTPUT_FOLDER = "output"

    print("=" * 60)
    print("      AveFace 纯算法测试程序（无GUI）")
    print("=" * 60)

    # 校验模型文件
    if not os.path.exists(MODEL_FILE):
        print(f"❌ 错误：未找到模型文件 {MODEL_FILE}")
        sys.exit(1)

    # 自动创建图片文件夹
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"📁 已自动创建文件夹: {IMAGE_FOLDER}")
        print("👉 请放入至少2张人脸图片，重新运行程序！")
        sys.exit(0)

    # 启动主流程
    run_average_face(IMAGE_FOLDER, MODEL_FILE, OUTPUT_FOLDER)