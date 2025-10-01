# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: run.py

import argparse
import datetime
import os
import pickle
import platform
import subprocess
import time

import cv2
import numpy as np
from colorama import Fore, Style
from omegaconf import OmegaConf
from tqdm import tqdm

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio

if platform.system().lower() == "windows":
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


# === ADJUSTABLE SMOOTHING STABILIZER ===
class AdjustableStabilizer:
    """Smoothing stabilizer with real-time adjustable parameters"""

    def __init__(
        self, alpha=0.3, movement_threshold=1.5
    ):  # FIXED: Consistent threshold
        self.alpha = alpha
        self.movement_threshold = movement_threshold
        self.prev_frame = None
        self.is_moving = False

    def stabilize(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame

        # Calculate movement
        movement = self.calculate_movement(self.prev_frame, frame)

        # Only apply smoothing when still
        if movement < self.movement_threshold:
            smoothed = cv2.addWeighted(
                frame, 1 - self.alpha, self.prev_frame, self.alpha, 0
            )
            self.prev_frame = smoothed.copy()
            self.is_moving = False
            return smoothed
        else:
            self.prev_frame = frame.copy()
            self.is_moving = True
            return frame

    def calculate_movement(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return np.mean(magnitude)

    def increase_smoothing(self):
        self.alpha = min(0.9, self.alpha + 0.1)
        print(f"Smoothing increased: alpha={self.alpha:.2f}")

    def decrease_smoothing(self):
        self.alpha = max(0.1, self.alpha - 0.1)
        print(f"Smoothing decreased: alpha={self.alpha:.2f}")

    def increase_threshold(self):
        self.movement_threshold = min(20.0, self.movement_threshold + 1.0)
        print(f"Movement threshold increased: {self.movement_threshold:.1f}")

    def decrease_threshold(self):
        self.movement_threshold = max(1.0, self.movement_threshold - 1.0)
        print(f"Movement threshold decreased: {self.movement_threshold:.1f}")


def run_with_video(args):
    print(
        Fore.RED
        + "Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate"
        + Style.RESET_ALL
    )
    print(
        Fore.GREEN
        + "1/2 > Smoothing,  3/4 > Movement Threshold,  0 > Show Settings"
        + Style.RESET_ALL
    )

    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    # Good settings for stability
    infer_cfg.infer_params.flag_relative_motion = True
    infer_cfg.infer_params.flag_stitching = True
    infer_cfg.infer_params.animation_region = "all"

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)

    # FORCE RESIZE FOR REALTIME PERFORMANCE
    if args.realtime:
        print("Optimizing for realtime performance...")
        # Create a temporary resized version if source is large
        original_img = cv2.imread(args.src_image)
        if original_img is not None and max(original_img.shape[:2]) > 512:
            print(
                f"Source image is large: {original_img.shape}, resizing to 512x512 for better performance"
            )
            temp_resized_path = "temp_resized_source.jpg"
            resized_img = cv2.resize(original_img, (512, 512))
            cv2.imwrite(temp_resized_path, resized_img)
            ret = pipe.prepare_source(temp_resized_path, realtime=args.realtime)
            # Clean up temp file
            try:
                os.remove(temp_resized_path)
            except:
                pass
        else:
            ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    else:
        ret = pipe.prepare_source(args.src_image, realtime=args.realtime)

    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)

    print(f"Source image size for processing: {pipe.src_imgs[0].shape}")

    # Initialize adjustable stabilizer - FIXED: Consistent threshold
    stabilizer = (
        AdjustableStabilizer(alpha=0.3, movement_threshold=1.5)
        if args.realtime
        else None
    )

    if args.dri_video and os.path.exists(args.dri_video):
        vcap = cv2.VideoCapture(args.dri_video)
    else:
        vcap = cv2.VideoCapture(0)
        if not vcap.isOpened():
            print("no camera found! exit!")
            exit(1)

    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vsave_crop_path = os.path.join(
            save_dir,
            f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4",
        )
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vsave_org_path = os.path.join(
            save_dir,
            f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4",
        )
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []

    frame_ind = 0

    # === FORCE PASTEBACK FOR REALTIME MODE ===
    if args.realtime:
        infer_cfg.infer_params.flag_pasteback = True
        infer_cfg.infer_params.flag_do_crop = True
        infer_cfg.infer_params.flag_stitching = True
        print("Forced pasteback enabled for realtime full output")

    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break
        t0 = time.time()
        first_frame = frame_ind == 0

        # FIXED: Correct parameter order for pipe.run()
        dri_crop, out_crop, out_org, dri_motion_info = pipe.run(
            frame,  # image (driving frame) FIRST
            pipe.src_imgs[0],  # img_src (source image) SECOND
            pipe.src_infos[0],  # src_info (source info) THIRD
            first_frame=first_frame,
        )

        frame_ind += 1
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        motion_lst.append(dri_motion_info[0])
        c_eyes_lst.append(dri_motion_info[1])
        c_lip_lst.append(dri_motion_info[2])

        infer_times.append(time.time() - t0)
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)

        # Apply stabilization
        if args.realtime and stabilizer:
            if infer_cfg.infer_params.flag_pasteback and out_org is not None:
                out_org = stabilizer.stabilize(out_org)
            else:
                out_crop = stabilizer.stabilize(out_crop)

        if not args.realtime:
            vout_crop.write(out_crop)
            if out_org is not None:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                vout_org.write(out_org)
        else:
            # FIXED: Safe display with fallback
            if out_org is not None:
                out_org_display = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow("Render", out_org_display)
            else:
                # Fallback to cropped view if full output isn't available
                cv2.imshow("Render", out_crop)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            # Existing keys
            if k == ord("s"):
                infer_cfg.infer_params.flag_stitching = (
                    not infer_cfg.infer_params.flag_stitching
                )
                print("flag_stitching:" + str(infer_cfg.infer_params.flag_stitching))
            if k == ord("z"):
                infer_cfg.infer_params.flag_relative_motion = (
                    not infer_cfg.infer_params.flag_relative_motion
                )
                print(
                    "flag_relative_motion:"
                    + str(infer_cfg.infer_params.flag_relative_motion)
                )
            if k == ord("x"):
                if infer_cfg.infer_params.animation_region == "all":
                    infer_cfg.infer_params.animation_region = "exp"
                    print('animation_region = "exp"')
                else:
                    infer_cfg.infer_params.animation_region = "all"
                    print('animation_region = "all"')
            if k == ord("c"):
                infer_cfg.infer_params.flag_crop_driving_video = (
                    not infer_cfg.infer_params.flag_crop_driving_video
                )
                print(
                    "flag_crop_driving_video:"
                    + str(infer_cfg.infer_params.flag_crop_driving_video)
                )
            # NEW SMOOTHING CONTROLS
            if k == ord("1"):  # Decrease smoothing (less blur, more jitter)
                if stabilizer:
                    stabilizer.decrease_smoothing()
            if k == ord("2"):  # Increase smoothing (more blur, less jitter)
                if stabilizer:
                    stabilizer.increase_smoothing()
            if k == ord("3"):  # Decrease movement threshold (smoother when moving)
                if stabilizer:
                    stabilizer.decrease_threshold()
            if k == ord("4"):  # Increase movement threshold (less smooth when moving)
                if stabilizer:
                    stabilizer.increase_threshold()
            if k == ord("0"):  # Print current settings
                if stabilizer:
                    print(
                        f"Current: alpha={stabilizer.alpha:.2f}, threshold={stabilizer.movement_threshold:.1f}"
                    )

    vcap.release()
    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [
                    FFMPEG,
                    "-i",
                    vsave_crop_path,
                    "-i",
                    args.dri_video,
                    "-b:v",
                    "10M",
                    "-c:v",
                    "libx264",
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:a",
                    "aac",
                    "-pix_fmt",
                    "yuv420p",
                    vsave_crop_path_new,
                    "-y",
                    "-shortest",
                ]
            )
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [
                    FFMPEG,
                    "-i",
                    vsave_org_path,
                    "-i",
                    args.dri_video,
                    "-b:v",
                    "10M",
                    "-c:v",
                    "libx264",
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:a",
                    "aac",
                    "-pix_fmt",
                    "yuv420p",
                    vsave_org_path_new,
                    "-y",
                    "-shortest",
                ]
            )
            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(
            np.median(infer_times) * 1000, np.mean(infer_times) * 1000
        )
    )

    template_dct = {
        "n_frames": len(motion_lst),
        "output_fps": fps,
        "motion": motion_lst,
        "c_eyes_lst": c_eyes_lst,
        "c_lip_lst": c_lip_lst,
    }
    template_pkl_path = os.path.join(
        save_dir, f"{os.path.basename(args.dri_video)}.pkl"
    )
    with open(template_pkl_path, "wb") as fw:
        pickle.dump(template_dct, fw)
    print(f"save driving motion pkl file at : {template_pkl_path}")


def run_with_pkl(args):
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    # Good settings for stability
    infer_cfg.infer_params.flag_relative_motion = True
    infer_cfg.infer_params.flag_stitching = True
    infer_cfg.infer_params.animation_region = "all"

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        return
    with open(args.dri_video, "rb") as fin:
        dri_motion_infos = pickle.load(fin)

    fps = int(dri_motion_infos["output_fps"])
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vsave_crop_path = os.path.join(
            save_dir,
            f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4",
        )
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512, 512))
        vsave_org_path = os.path.join(
            save_dir,
            f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4",
        )
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = dri_motion_infos["motion"]
    c_eyes_lst = (
        dri_motion_infos["c_eyes_lst"]
        if "c_eyes_lst" in dri_motion_infos
        else dri_motion_infos["c_d_eyes_lst"]
    )
    c_lip_lst = (
        dri_motion_infos["c_lip_lst"]
        if "c_lip_lst" in dri_motion_infos
        else dri_motion_infos["c_d_lip_lst"]
    )

    frame_num = len(motion_lst)
    stabilizer = (
        AdjustableStabilizer(alpha=0.3, movement_threshold=1.5)
        if args.realtime
        else None
    )

    # === FORCE PASTEBACK FOR REALTIME MODE ===
    if args.realtime:
        infer_cfg.infer_params.flag_pasteback = True
        infer_cfg.infer_params.flag_do_crop = True
        infer_cfg.infer_params.flag_stitching = True
        print("Forced pasteback enabled for realtime full output")

    for frame_ind in tqdm(range(frame_num)):
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_motion_info_ = [
            motion_lst[frame_ind],
            c_eyes_lst[frame_ind],
            c_lip_lst[frame_ind],
        ]
        out_crop, out_org = pipe.run_with_pkl(
            dri_motion_info_,
            pipe.src_imgs[0],
            pipe.src_infos[0],
            first_frame=first_frame,
        )
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        infer_times.append(time.time() - t0)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)

        if args.realtime and stabilizer:
            if infer_cfg.infer_params.flag_pasteback and out_org is not None:
                out_org = stabilizer.stabilize(out_org)
            else:
                out_crop = stabilizer.stabilize(out_crop)

        if not args.realtime:
            vout_crop.write(out_crop)
            if out_org is not None:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                vout_org.write(out_org)
        else:
            # FIXED: Safe display with fallback
            if out_org is not None:
                out_org_display = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow(
                    "Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate",
                    out_org_display,
                )
            else:
                cv2.imshow(
                    "Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate",
                    out_crop,
                )

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            # Existing keys
            if k == ord("s"):
                infer_cfg.infer_params.flag_stitching = (
                    not infer_cfg.infer_params.flag_stitching
                )
                print("flag_stitching:" + str(infer_cfg.infer_params.flag_stitching))
            if k == ord("z"):
                infer_cfg.infer_params.flag_relative_motion = (
                    not infer_cfg.infer_params.flag_relative_motion
                )
                print(
                    "flag_relative_motion:"
                    + str(infer_cfg.infer_params.flag_relative_motion)
                )
            if k == ord("x"):
                if infer_cfg.infer_params.animation_region == "all":
                    infer_cfg.infer_params.animation_region = "exp"
                    print('animation_region = "exp"')
                else:
                    infer_cfg.infer_params.animation_region = "all"
                    print('animation_region = "all"')
            if k == ord("c"):
                infer_cfg.infer_params.flag_crop_driving_video = (
                    not infer_cfg.infer_params.flag_crop_driving_video
                )
                print(
                    "flag_crop_driving_video:"
                    + str(infer_cfg.infer_params.flag_crop_driving_video)
                )
            # NEW SMOOTHING CONTROLS
            if k == ord("1"):  # Decrease smoothing (less blur, more jitter)
                if stabilizer:
                    stabilizer.decrease_smoothing()
            if k == ord("2"):  # Increase smoothing (more blur, less jitter)
                if stabilizer:
                    stabilizer.increase_smoothing()
            if k == ord("3"):  # Decrease movement threshold (smoother when moving)
                if stabilizer:
                    stabilizer.decrease_threshold()
            if k == ord("4"):  # Increase movement threshold (less smooth when moving)
                if stabilizer:
                    stabilizer.increase_threshold()
            if k == ord("0"):  # Print current settings
                if stabilizer:
                    print(
                        f"Current: alpha={stabilizer.alpha:.2f}, threshold={stabilizer.movement_threshold:.1f}"
                    )

    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [
                    FFMPEG,
                    "-i",
                    vsave_crop_path,
                    "-i",
                    args.dri_video,
                    "-b:v",
                    "10M",
                    "-c:v",
                    "libx264",
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:a",
                    "aac",
                    "-pix_fmt",
                    "yuv420p",
                    vsave_crop_path_new,
                    "-y",
                    "-shortest",
                ]
            )
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [
                    FFMPEG,
                    "-i",
                    vsave_org_path,
                    "-i",
                    args.dri_video,
                    "-b:v",
                    "10M",
                    "-c:v",
                    "libx264",
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c:a",
                    "aac",
                    "-pix_fmt",
                    "yuv420p",
                    vsave_org_path_new,
                    "-y",
                    "-shortest",
                ]
            )
            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(
            np.median(infer_times) * 1000, np.mean(infer_times) * 1000
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster Live Portrait Pipeline")
    parser.add_argument(
        "--src_image",
        required=False,
        type=str,
        default="assets/examples/source/s12.jpg",
        help="source image",
    )
    parser.add_argument(
        "--dri_video",
        required=False,
        type=str,
        default="assets/examples/driving/d14.mp4",
        help="driving video",
    )
    parser.add_argument(
        "--cfg",
        required=False,
        type=str,
        default="configs/onnx_infer.yaml",
        help="inference config",
    )
    parser.add_argument("--realtime", action="store_true", help="realtime inference")
    parser.add_argument("--animal", action="store_true", help="use animal model")
    parser.add_argument(
        "--paste_back",
        action="store_true",
        default=False,
        help="paste back to origin image",
    )
    args, unknown = parser.parse_known_args()

    if args.dri_video.endswith(".pkl"):
        run_with_pkl(args)
    else:
        run_with_video(args)
