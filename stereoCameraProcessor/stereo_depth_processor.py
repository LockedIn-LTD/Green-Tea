#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from glob import glob
from typing import Tuple, Optional

import numpy as np
import cv2

# Optional VPI
try:
    import vpi
    HAS_VPI = True
except Exception:
    HAS_VPI = False


# ---------------------------
# Calibration parsing
# ---------------------------

def _extract_floats_after(label: str, text: str, count: Optional[int] = None) -> np.ndarray:
    lbl = label.lower()
    if lbl not in text.lower():
        raise ValueError(f"Label '{label}' not found in file")
    start = text.lower().index(lbl) + len(label)
    chunk = text[start:]
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', chunk)
    if count is not None:
        if len(nums) < count:
            raise ValueError(f"Expected at least {count} numbers after '{label}', got {len(nums)}")
        nums = nums[:count]
    return np.array([float(x) for x in nums], dtype=np.float64)


def load_intrinsics_dat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        text = f.read()
    K_vals = _extract_floats_after("intrinsic:", text, count=9)
    K = K_vals.reshape(3, 3)
    D_vals = _extract_floats_after("distortion:", text, count=None)
    D = np.array(D_vals, dtype=np.float64).reshape(-1, 1)
    return K, D


def load_extrinsics_dat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        text = f.read()
    R_vals = _extract_floats_after("R:", text, count=9)
    T_vals = _extract_floats_after("T:", text, count=3)
    R = R_vals.reshape(3, 3)
    T = T_vals.reshape(3, 1)
    return R, T


# ---------------------------
# GStreamer helpers (Jetson)
# ---------------------------

def gst_pipeline(sensor_id=0,
                 capture_width=1280,
                 capture_height=720,
                 display_width=None,
                 display_height=None,
                 framerate=30,
                 flip_method=2,  # default to 2 per your request
                 exposure_time_us: Optional[int] = None,
                 gain: Optional[float] = None) -> str:
    display_width = display_width or capture_width
    display_height = display_height or capture_height
    exposure_str = "" if exposure_time_us is None else f" exposuretimerange='{exposure_time_us} {exposure_time_us}'"
    gain_str = "" if gain is None else f" gainrange='{gain} {gain}'"
    return (
        f"nvarguscamerasrc sensor-id={sensor_id}{exposure_str}{gain_str} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
    )


def open_camera(sensor_id: int, width: int, height: int, fps: int, flip: int) -> cv2.VideoCapture:
    pipe = gst_pipeline(sensor_id=sensor_id, capture_width=width, capture_height=height,
                        display_width=width, display_height=height, framerate=fps, flip_method=flip)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera sensor-id={sensor_id} (flip={flip})")
    return cap


# ---------------------------
# Stereo Processor
# ---------------------------

class StereoDepthProcessor:
    def __init__(self,
                 calib_dir: str = "camera_parameters",
                 image_size: Tuple[int, int] = (1920, 1080),
                 algo: str = "auto",            # auto | cuda_bm | vpi | cpu_sgbm
                 max_disparity: int = 128,      # multiple of 16
                 block_size: int = 15,          # odd >=5
                 alpha: float = 0.0,            # 0..1 (stereoRectify crop vs keep)
                 use_cuda_rectify: bool = False):
        self.calib_dir = calib_dir
        self.size = image_size
        self.max_disp = int(np.ceil(max_disparity / 16) * 16)
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.use_cuda_rectify = bool(use_cuda_rectify)

        self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if algo == "auto":
            if self.has_cuda:
                self.algo = "cuda_bm"
            elif HAS_VPI:
                self.algo = "vpi"
            else:
                self.algo = "cpu_sgbm"
        else:
            self.algo = algo

        self._load_calibration()
        self._create_rectify_maps()
        self._init_matchers()

        print(f"[StereoDepthProcessor] algo={self.algo}, size={self.size}, "
              f"max_disp={self.max_disp}, block_size={self.block_size}, "
              f"fx={self.fx:.2f}, baseline={self.baseline_m:.6f} m")

    # --- Calibration and rectification ---

    def _load_calibration(self):
        cam0_intr = os.path.join(self.calib_dir, "camera0_intrinsics.dat")
        cam1_intr = os.path.join(self.calib_dir, "camera1_intrinsics.dat")
        cam1_ext = os.path.join(self.calib_dir, "camera1_rot_trans.dat")
        if not (os.path.isfile(cam0_intr) and os.path.isfile(cam1_intr) and os.path.isfile(cam1_ext)):
            raise FileNotFoundError("Missing calibration files in camera_parameters/")

        self.K0, self.D0 = load_intrinsics_dat(cam0_intr)
        self.K1, self.D1 = load_intrinsics_dat(cam1_intr)
        self.R01, self.T01 = load_extrinsics_dat(cam1_ext)

        # mm -> m if needed (your T ≈ 60 mm)
        if float(np.linalg.norm(self.T01)) > 1.0:
            self.T01 = self.T01 / 1000.0

        w, h = self.size
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(
            self.K0, self.D0, self.K1, self.D1, (w, h), self.R01, self.T01, alpha=self.alpha,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

        self.fx = float(self.P1[0, 0])
        # Ensure positive baseline magnitude so depths are positive.
        self.baseline_m = abs(float(-self.P2[0, 3] / self.fx))

    def _create_rectify_maps(self):
        w, h = self.size
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(
            self.K0, self.D0, self.R1, self.P1, (w, h), cv2.CV_32FC1
        )
        self.map2_x, self.map2_y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R2, self.P2, (w, h), cv2.CV_32FC1
        )

        self.gpu_maps_ready = False
        if self.use_cuda_rectify and self.has_cuda:
            self.gpu_map1_x = cv2.cuda_GpuMat(self.map1_x)
            self.gpu_map1_y = cv2.cuda_GpuMat(self.map1_y)
            self.gpu_map2_x = cv2.cuda_GpuMat(self.map2_x)
            self.gpu_map2_y = cv2.cuda_GpuMat(self.map2_y)
            self.gpu_maps_ready = True

    def _init_matchers(self):
        self.cuda_bm = None
        self.cpu_sgbm = None
        self.vpi_estimator = None
        self.vpi_stream = None
        self.cuda_stream = None

        if self.algo == "cuda_bm":
            if not self.has_cuda:
                raise RuntimeError("Requested CUDA BM, but CUDA is not available.")
            self.cuda_bm = cv2.cuda.createStereoBM(numDisparities=self.max_disp,
                                                   blockSize=self.block_size)
            self.cuda_bm.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
            self.cuda_bm.setPreFilterSize(9)
            self.cuda_bm.setPreFilterCap(31)
            self.cuda_bm.setMinDisparity(0)
            self.cuda_bm.setTextureThreshold(10)
            self.cuda_bm.setUniquenessRatio(10)
            self.cuda_bm.setSpeckleWindowSize(100)
            self.cuda_bm.setSpeckleRange(32)
            self.cuda_stream = cv2.cuda.Stream()

        elif self.algo == "cpu_sgbm":
            P1 = 8 * 1 * (self.block_size ** 2)
            P2 = 32 * 1 * (self.block_size ** 2)
            self.cpu_sgbm = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.max_disp,
                blockSize=self.block_size,
                P1=P1, P2=P2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

        elif self.algo == "vpi":
            if not HAS_VPI:
                raise RuntimeError("Requested VPI, but VPI Python bindings not available.")
            w, h = self.size
            self.vpi_stream = vpi.Stream()
            self.vpi_estimator = vpi.StereoDisparityEstimator((w, h),
                                                              window=self.block_size,
                                                              maxdisp=self.max_disp)

    # --- Processing ---

    def _rectify_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        if self.use_cuda_rectify and self.has_cuda and self.gpu_maps_ready:
            gl = cv2.cuda_GpuMat(); gr = cv2.cuda_GpuMat()
            gl.upload(left_bgr); gr.upload(right_bgr)
            try:
                rl = cv2.cuda.remap(gl, self.gpu_map1_x, self.gpu_map1_y, interpolation=cv2.INTER_LINEAR)
                rr = cv2.cuda.remap(gr, self.gpu_map2_x, self.gpu_map2_y, interpolation=cv2.INTER_LINEAR)
            except TypeError:
                rl = cv2.cuda.remap(gl, self.gpu_map1_x, self.gpu_map1_y, cv2.INTER_LINEAR,
                                    cv2.BORDER_CONSTANT, (0,0,0,0), None, self.cuda_stream)
                rr = cv2.cuda.remap(gr, self.gpu_map2_x, self.gpu_map2_y, cv2.INTER_LINEAR,
                                    cv2.BORDER_CONSTANT, (0,0,0,0), None, self.cuda_stream)
                self.cuda_stream.waitForCompletion()
            left_rect = rl.download(); right_rect = rr.download()
        else:
            left_rect = cv2.remap(left_bgr, self.map1_x, self.map1_y, interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_bgr, self.map2_x, self.map2_y, interpolation=cv2.INTER_LINEAR)
        return left_rect, right_rect

    def _bm_cuda_disparity(self, left_rect_bgr: np.ndarray, right_rect_bgr: np.ndarray) -> np.ndarray:
        gl = cv2.cuda_GpuMat(); gr = cv2.cuda_GpuMat()
        gl.upload(left_rect_bgr); gr.upload(right_rect_bgr)
        gl_gray = cv2.cuda.cvtColor(gl, cv2.COLOR_BGR2GRAY)
        gr_gray = cv2.cuda.cvtColor(gr, cv2.COLOR_BGR2GRAY)
        stream = self.cuda_stream if self.cuda_stream is not None else cv2.cuda.Stream()
        try:
            disp_gpu = self.cuda_bm.compute(gl_gray, gr_gray, stream)
            stream.waitForCompletion()
        except Exception:
            disp_gpu = cv2.cuda_GpuMat()
            self.cuda_bm.compute(gl_gray, gr_gray, disp_gpu, stream)
            stream.waitForCompletion()
        disp = disp_gpu.download().astype(np.float32) / 16.0
        return disp

    def _sgbm_cpu_disparity(self, left_rect_bgr: np.ndarray, right_rect_bgr: np.ndarray) -> np.ndarray:
        l_gray = cv2.cvtColor(left_rect_bgr, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(right_rect_bgr, cv2.COLOR_BGR2GRAY)
        disp = self.cpu_sgbm.compute(l_gray, r_gray).astype(np.float32) / 16.0
        return disp

    def _vpi_disparity(self, left_rect_bgr: np.ndarray, right_rect_bgr: np.ndarray) -> np.ndarray:
        l_gray = cv2.cvtColor(left_rect_bgr, cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(right_rect_bgr, cv2.COLOR_BGR2GRAY)
        with vpi.Backend.CUDA:
            li = vpi.asimage(l_gray); ri = vpi.asimage(r_gray)
            disp_vpi = self.vpi_estimator.submit(self.vpi_stream, li, ri)
            self.vpi_stream.sync()
            disp_f32 = disp_vpi.convert(vpi.Format.F32).cpu()
            disp = np.array(disp_f32, copy=False)
        return disp

    def disparity_to_depth(self, disparity: np.ndarray, min_disp: float = 1.0) -> np.ndarray:
        disp = disparity.copy()
        disp[disp < min_disp] = np.nan
        depth = self.fx * self.baseline_m / disp
        return depth

    def process_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        left_rect, right_rect = self._rectify_pair(left_bgr, right_bgr)
        if self.algo == "cuda_bm":
            disp = self._bm_cuda_disparity(left_rect, right_rect)
        elif self.algo == "vpi":
            disp = self._vpi_disparity(left_rect, right_rect)
        else:
            disp = self._sgbm_cpu_disparity(left_rect, right_rect)
        depth = self.disparity_to_depth(disp)
        return left_rect, right_rect, disp, depth

    # Visualization helpers

    @staticmethod
    def colorize_disparity(disp: np.ndarray, max_disp_vis: Optional[float] = None) -> np.ndarray:
        m = np.nanmax(disp) if max_disp_vis is None else max_disp_vis
        if not np.isfinite(m) or m <= 0: m = 1.0
        norm = np.clip(disp / m, 0, 1)
        norm = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    @staticmethod
    def colorize_depth(depth_m: np.ndarray, max_depth_m: float = 5.0) -> np.ndarray:
        d = depth_m.copy()
        d[~np.isfinite(d)] = 0
        d = np.clip(d, 0, max_depth_m)
        inv = (1.0 - (d / max_depth_m))  # nearer => hotter
        inv = (inv * 255).astype(np.uint8)
        return cv2.applyColorMap(inv, cv2.COLORMAP_TURBO)


# ---------------------------
# CLI runners
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--calib-dir", type=str, default="camera_parameters")
    ap.add_argument("--input", type=str, choices=["camera", "images"], default="camera")
    ap.add_argument("--images-dir", type=str, default="frames_pair")
    ap.add_argument("--left-id", type=int, default=0)
    ap.add_argument("--right-id", type=int, default=1)
    ap.add_argument("--flip", type=int, default=2, help="nvvidconv flip-method for both sensors")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--algo", type=str, choices=["auto", "cuda_bm", "vpi", "cpu_sgbm"], default="auto")
    ap.add_argument("--max-disp", type=int, default=224)
    ap.add_argument("--block-size", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--use-cuda-rectify", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--heatmap", type=str, choices=["depth", "disparity"], default="depth")
    ap.add_argument("--display-scale", type=float, default=0.5, help="Resize factor for on-screen preview")
    ap.add_argument("--max-depth-vis", type=float, default=5.0, help="Depth colormap max (m)")
    ap.add_argument("--face", action="store_true", help="Estimate face distance using Haar cascade")
    return ap.parse_args()


def load_face_cascade():
    # 1. Define the specific, verified path for your Jetson/NVIDIA setup
    JETSON_CASCADE_DIR = "/usr/share/opencv4/haarcascades/"
    FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"
    
    # Check the Jetson path first
    jetson_path = os.path.join(JETSON_CASCADE_DIR, FACE_CASCADE_FILE)
    if os.path.isfile(jetson_path):
        print(f"Loading cascade from: {jetson_path}")
        return cv2.CascadeClassifier(jetson_path)
    
    # 2. Fallback to the standard OpenCV data path (for general compatibility)
    try:
        # NOTE: cv2.data.haarcascades already includes a trailing slash
        standard_path = cv2.data.haarcascades + FACE_CASCADE_FILE
        if os.path.isfile(standard_path):
            print(f"Loading cascade from standard path: {standard_path}")
            return cv2.CascadeClassifier(standard_path)
    except AttributeError:
        # cv2.data.haarcascades might not exist in older or specialized builds
        pass
    except Exception as e:
        print(f"Error loading cascade from standard path: {e}")
        pass
        
    print("WARNING: Face cascade file was not found in either custom or standard location.")
    return None


def overlay_face_distance(heatmap_bgr: np.ndarray,
                          left_rect_bgr: np.ndarray,
                          depth_m: np.ndarray,
                          face_cascade,
                          ema_state: dict,
                          max_depth_m: float) -> None:
    if face_cascade is None:
        cv2.putText(heatmap_bgr, "Face detect: cascade not found", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return

    gray = cv2.cvtColor(left_rect_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                          minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        cv2.putText(heatmap_bgr, "Face: not found", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return

    # Choose largest face
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    x1, y1 = max(x,0), max(y,0)
    x2, y2 = min(x+w, depth_m.shape[1]-1), min(y+h, depth_m.shape[0]-1)

    roi = depth_m[y1:y2, x1:x2]
    valid = np.isfinite(roi)
    valid_ratio = float(valid.sum()) / max(1, roi.size)

    dist = np.nan
    if valid_ratio > 0.1:
        dist = float(np.nanmedian(roi))

    # Draw box on heatmap
    cv2.rectangle(heatmap_bgr, (x1, y1), (x2, y2), (0,255,0), 2)

    # Smooth the readout for stability
    if np.isfinite(dist):
        if "ema" not in ema_state:
            ema_state["ema"] = dist
        else:
            ema_state["ema"] = 0.7 * ema_state["ema"] + 0.3 * dist

    text = "Face: -- m"
    if "ema" in ema_state and np.isfinite(ema_state["ema"]):
        text = f"Face: {ema_state['ema']:.2f} m"

    cv2.putText(heatmap_bgr, text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(heatmap_bgr, text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Add a small legend for depth heatmap
    legend = f"Heatmap range: 0..{max_depth_m:.1f} m (hot=near)"
    cv2.putText(heatmap_bgr, legend, (12, heatmap_bgr.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(heatmap_bgr, legend, (12, heatmap_bgr.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)


def show_heatmap_window(vis_bgr: np.ndarray, scale: float, win_name: str = "Depth Heatmap"):
    if scale != 1.0:
        h, w = vis_bgr.shape[:2]
        vis_bgr = cv2.resize(vis_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win_name, vis_bgr)


def run_camera(args):
    proc = StereoDepthProcessor(
        calib_dir=args.calib_dir,
        image_size=(args.width, args.height),
        algo=args.algo,
        max_disparity=args.max_disp,
        block_size=args.block_size,
        alpha=args.alpha,
        use_cuda_rectify=args.use_cuda_rectify
    )

    capL = open_camera(args.left_id, args.width, args.height, args.fps, args.flip)
    capR = open_camera(args.right_id, args.width, args.height, args.fps, args.flip)

    face_cascade = load_face_cascade() if args.face else None
    ema_state = {}

    print("Press 'q' to quit.")
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not (retL and retR):
            print("Camera read failed; retrying...")
            continue

        left_rect, right_rect, disp, depth = proc.process_pair(frameL, frameR)

        # Heatmap choice
        if args.heatmap == "depth":
            vis = proc.colorize_depth(depth, max_depth_m=args.max_depth_vis)
        else:
            vis = proc.colorize_disparity(disp, max_disp_vis=args.max_disp)

        # Face distance overlay (uses left rect + depth)
        if args.face and args.heatmap == "depth":
            overlay_face_distance(vis, left_rect, depth, face_cascade, ema_state, args.max_depth_vis)

        if args.show:
            show_heatmap_window(vis, args.display_scale)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capL.release(); capR.release()
    cv2.destroyAllWindows()


def run_images(args):
    proc = StereoDepthProcessor(
        calib_dir=args.calib_dir,
        image_size=(args.width, args.height),
        algo=args.algo,
        max_disparity=args.max_disp,
        block_size=args.block_size,
        alpha=args.alpha,
        use_cuda_rectify=args.use_cuda_rectify
    )
    left_paths = sorted(glob(os.path.join(args.images_dir, "left_*.*")))
    right_paths = sorted(glob(os.path.join(args.images_dir, "right_*.*")))
    if len(left_paths) == 0 or len(left_paths) != len(right_paths):
        print("No image pairs or mismatch in counts. Expected left_*.png and right_*.png.")
        return

    face_cascade = load_face_cascade() if args.face else None
    ema_state = {}

    for lp, rp in zip(left_paths, right_paths):
        L = cv2.imread(lp, cv2.IMREAD_COLOR)
        R = cv2.imread(rp, cv2.IMREAD_COLOR)
        if L is None or R is None:
            print(f"Failed to read: {lp}, {rp}")
            continue
        if (L.shape[1], L.shape[0]) != (args.width, args.height):
            L = cv2.resize(L, (args.width, args.height), interpolation=cv2.INTER_AREA)
        if (R.shape[1], R.shape[0]) != (args.width, args.height):
            R = cv2.resize(R, (args.width, args.height), interpolation=cv2.INTER_AREA)

        left_rect, right_rect, disp, depth = proc.process_pair(L, R)

        if args.heatmap == "depth":
            vis = proc.colorize_depth(depth, max_depth_m=args.max_depth_vis)
        else:
            vis = proc.colorize_disparity(disp, max_disp_vis=args.max_disp)

        if args.face and args.heatmap == "depth":
            overlay_face_distance(vis, left_rect, depth, face_cascade, ema_state, args.max_depth_vis)

        if args.show:
            show_heatmap_window(vis, args.display_scale)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.input == "camera":
        run_camera(args)
    else:
        run_images(args)


if __name__ == "__main__":
    main()