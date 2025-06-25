import base64
import json
import logging
import os
import shutil
import subprocess
import uuid
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Correctly import the MedSAM2 video predictor
# Ensure 'sam2' library is correctly installed and accessible in your environment
# For Windows, you might need to install specific PyTorch versions with CUDA support
# and ensure your NVIDIA drivers are up to date.
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.amg import mask_to_rle_pytorch, rle_to_mask

# --- Basic Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MedSAM2 Video Segmentation API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- Configuration ---
class AppConfig:
    def __init__(self):
        self.downsample_factor = 1
        self.framerate = 24
        self.mask_opacity = 0.6
        self.save_masks = True
        self.save_videos = True
        self.clip_duration = 30  # seconds for long-form video segmentation
        self.border_thickness = 2
        self.colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203),  # Pink
            (128, 255, 0),  # Lime
            (255, 165, 0),  # Dark Orange
            (0, 128, 255),  # Sky Blue
            (255, 69, 0),  # Red Orange
            (50, 205, 50),  # Lime Green
            (255, 20, 147),  # Deep Pink
            (0, 191, 255),  # Deep Sky Blue
            (255, 215, 0),  # Gold
            (138, 43, 226),  # Blue Violet
            (220, 20, 60),  # Crimson
            (32, 178, 170),  # Light Sea Green
        ]


config = AppConfig()

# --- Model and Session Management ---
session_states = {}

try:
    # IMPORTANT: Update these paths to your MedSAM2 model and config file
    # For Windows, ensure these paths use '/' or are correctly escaped if '\' is used directly
    # e.g., "C:/Users/YourUser/MedSAM2/checkpoints/MedSAM2_latest.pt"
    checkpoint = "MedSAM2/checkpoints/MedSAM2_latest.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t512.yaml"

    # device="cuda" will work if CUDA is properly set up on Windows.
    # If not, it will raise an error, or you can explicitly set device="cpu" for testing.
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")
    logger.info("Successfully loaded MedSAM2 model on CUDA device.")
except Exception as e:
    logger.error(f"Failed to load MedSAM2 model: {e}")
    logger.error(
        "Please ensure the checkpoint and model_cfg paths are correct and the model files exist."
    )
    logger.error(
        "On Windows, also ensure CUDA is properly installed and configured for PyTorch."
    )
    predictor = None


# --- Pydantic Models for API Data Validation ---
class CreateSessionData(BaseModel):
    s3_link: str


class CreateSessionResponse(BaseModel):
    session_id: str
    frames: List[str]
    total_clips: Optional[int] = None


class ClickData(BaseModel):
    sessionId: str
    frameIndex: int
    objectId: int
    labels: List[int]
    points: List[List[float]]
    clearOldPoints: bool
    resetState: bool


class PropagateData(BaseModel):
    sessionId: str
    start_frame_index: int


class GenerateData(BaseModel):
    sessionId: str
    effect: str


class LongFormSegmentData(BaseModel):
    sessionId: str
    clipIndex: int
    objectId: int
    labels: List[int]
    points: List[List[float]]


class ConfigUpdateData(BaseModel):
    downsample_factor: Optional[int] = None
    framerate: Optional[int] = None
    mask_opacity: Optional[float] = None
    save_masks: Optional[bool] = None
    save_videos: Optional[bool] = None
    clip_duration: Optional[int] = None
    border_thickness: Optional[int] = None


# --- Utility Functions ---
def save_mask_metadata(
    session_id: str, frame_idx: int, object_id: int, color: tuple, label: str = None
):
    """Save mask metadata including color and label information"""
    session = session_states.get(session_id)
    if not session:
        return

    metadata_dir = os.path.join(session["frames_dir"], "masks_metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    metadata_file = os.path.join(
        metadata_dir, f"frame_{frame_idx:03d}_obj_{object_id}.json"
    )
    metadata = {
        "frame_index": frame_idx,
        "object_id": object_id,
        "color": {
            "rgb": color,
            "hex": "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2]),
        },
        "label": label or f"Object_{object_id}",
        "timestamp": frame_idx / config.framerate if config.framerate > 0 else 0,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def save_mask_images(session_id: str, frame_idx: int, masks_data: List[Dict]):
    """Save individual mask images as PNG files"""
    session = session_states.get(session_id)
    if not session or not config.save_masks:
        return

    frames_dir = session["frames_dir"]
    masks_dir = os.path.join(frames_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if frame_idx >= len(frame_files):
        return

    frame_filename = frame_files[frame_idx]
    input_image_path = os.path.join(frames_dir, frame_filename)
    input_image = cv2.imread(input_image_path)

    if input_image is None:
        return

    for mask_data in masks_data:
        object_id = mask_data["objectId"]
        rle_list = mask_data["rleMask"]

        # Create combined mask
        combined_mask = None
        for rle_dict in rle_list:
            single_mask = rle_to_mask(rle_dict)
            single_mask_array = np.array(single_mask).reshape(input_image.shape[:2])
            if combined_mask is None:
                combined_mask = single_mask_array
            else:
                combined_mask = np.logical_or(combined_mask, single_mask_array)

        if combined_mask is not None:
            # Save binary mask
            mask_filename = f"frame_{frame_idx:03d}_obj_{object_id}_mask.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(mask_path, (combined_mask * 255).astype(np.uint8))

            # Save colored mask overlay
            overlay = np.zeros_like(input_image)
            color = config.colors[object_id % len(config.colors)]
            overlay[combined_mask.astype(bool)] = color

            colored_mask_filename = f"frame_{frame_idx:03d}_obj_{object_id}_colored.png"
            colored_mask_path = os.path.join(masks_dir, colored_mask_filename)
            cv2.imwrite(colored_mask_path, overlay)


def clear_gpu_cache():
    """Clear GPU cache to prevent memory issues"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


def extract_last_frame_mask(session_id: str, clip_index: int) -> Optional[Dict]:
    """Extract mask from the last frame of a clip for continuity"""
    session = session_states.get(session_id)
    if not session or "clips" not in session:
        return None

    if clip_index == 0:
        return None

    prev_clip_results = (
        session.get("clips", {}).get(clip_index - 1, {}).get("results", {})
    )
    if not prev_clip_results:
        return None

    # Get the last frame's mask from previous clip
    last_frame_idx = max(prev_clip_results.keys()) if prev_clip_results else None
    if last_frame_idx is not None:
        return prev_clip_results[last_frame_idx]

    return None


# --- API Endpoints ---


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_memory = None

    if gpu_available:
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }

    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "gpu_available": gpu_available,
        "gpu_memory": gpu_memory,
        "active_sessions": len(session_states),
        "version": "2.0.0",
    }


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "downsample_factor": config.downsample_factor,
        "framerate": config.framerate,
        "mask_opacity": config.mask_opacity,
        "save_masks": config.save_masks,
        "save_videos": config.save_videos,
        "clip_duration": config.clip_duration,
        "border_thickness": config.border_thickness,
        "available_colors": len(config.colors),
    }


@app.post("/config")
async def update_config(data: ConfigUpdateData):
    """Update configuration settings"""
    updated_fields = []

    if data.downsample_factor is not None:
        config.downsample_factor = max(1, data.downsample_factor)
        updated_fields.append("downsample_factor")

    if data.framerate is not None:
        config.framerate = max(1, data.framerate)
        updated_fields.append("framerate")

    if data.mask_opacity is not None:
        config.mask_opacity = max(0.0, min(1.0, data.mask_opacity))
        updated_fields.append("mask_opacity")

    if data.save_masks is not None:
        config.save_masks = data.save_masks
        updated_fields.append("save_masks")

    if data.save_videos is not None:
        config.save_videos = data.save_videos
        updated_fields.append("save_videos")

    if data.clip_duration is not None:
        config.clip_duration = max(5, data.clip_duration)  # Minimum 5 seconds
        updated_fields.append("clip_duration")

    if data.border_thickness is not None:
        config.border_thickness = max(1, data.border_thickness)
        updated_fields.append("border_thickness")

    return {
        "message": "Configuration updated successfully",
        "updated_fields": updated_fields,
        "current_config": await get_config(),
    }


@app.on_event("startup")
async def startup_event():
    if predictor is None:
        raise RuntimeError(
            "MedSAM2 predictor could not be initialized. Check model paths and dependencies."
        )

    # Ensure the 'tmp' directory is created. This will work on Windows.
    os.makedirs("./tmp", exist_ok=True)
    logger.info("Application startup complete. Model loaded and tmp directory ensured.")


@app.post("/create_session_upload/", response_model=CreateSessionResponse)
async def create_session_upload(
    video_file: UploadFile = File(...),
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    downsample_factor: Optional[int] = None,
    long_form: Optional[bool] = False,
):
    session_id = str(uuid.uuid4())
    # Use os.path.join for cross-platform path construction
    frames_dir = os.path.join("./tmp", session_id)

    try:
        if not video_file.content_type or not video_file.content_type.startswith(
            "video/"
        ):
            raise HTTPException(status_code=400, detail="File must be a video.")

        os.makedirs(frames_dir, exist_ok=True)
        video_path = os.path.join(frames_dir, f"{session_id}.mp4")

        with open(video_path, "wb") as f:
            content = await video_file.read()
            f.write(content)

        # Get video information
        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        logger.info(
            f"[{session_id}] Video info: {original_width}x{original_height}, {fps}fps, {duration:.2f}s"
        )

        downsample = downsample_factor or config.downsample_factor

        if long_form and duration > config.clip_duration:
            # Handle long-form video segmentation
            total_clips = int(np.ceil(duration / config.clip_duration))
            logger.info(
                f"[{session_id}] Long-form video detected. Creating {total_clips} clips."
            )

            session_states[session_id] = {
                "frames_dir": frames_dir,
                "original_width": original_width,
                "original_height": original_height,
                "fps": fps,
                "duration": duration,
                "total_clips": total_clips,
                "long_form": True,
                "clips": {},
                "video_path": video_path,
            }

            return CreateSessionResponse(
                session_id=session_id,
                frames=[],  # Frames will be extracted per clip
                total_clips=total_clips,
            )
        else:
            # Handle regular video processing
            logger.info(
                f"[{session_id}] Regular video processing. Extracting frames..."
            )

            ffmpeg_command = ["ffmpeg", "-i", video_path]
            if start_time:
                ffmpeg_command.extend(["-ss", start_time])
            if end_time:
                ffmpeg_command.extend(["-to", end_time])

            video_filters = [f"fps={config.framerate}"]
            if downsample > 1:
                video_filters.append(f"scale=iw/{downsample}:ih/{downsample}")
                logger.info(f"[{session_id}] Downsampling by factor of {downsample}")

            ffmpeg_command.extend(["-vf", ",".join(video_filters)])
            # Changed pixel format to yuv420p for broader compatibility on Windows
            ffmpeg_command.extend(
                [
                    "-q:v",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    os.path.join(frames_dir, "%03d.jpg"),
                ]
            )

            # Using shell=True for ffmpeg on Windows can sometimes resolve issues
            # if ffmpeg is not directly in PATH, but generally not recommended for security.
            # If you encounter issues with ffmpeg not being found, try adding `shell=True` here:
            # subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, shell=True)
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            os.remove(video_path)

            logger.info(f"[{session_id}] Frame extraction complete.")

            inference_state = predictor.init_state(frames_dir)
            session_states[session_id] = {
                "inference_state": inference_state,
                "frames_dir": frames_dir,
                "results": {},
                "original_width": original_width,
                "original_height": original_height,
                "long_form": False,
            }

            frames = []
            for filename in sorted(os.listdir(frames_dir)):
                if filename.endswith(".jpg"):
                    with open(os.path.join(frames_dir, filename), "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )
                        frames.append(encoded_string)

            return CreateSessionResponse(session_id=session_id, frames=frames)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for session {session_id}: {e.stderr}")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        raise HTTPException(
            status_code=500, detail=f"Error processing video: {e.stderr}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in create_session_upload for session {session_id}: {e}"
        )
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_clip_session/")
async def create_clip_session(session_id: str, clip_index: int):
    """Extract frames for a specific clip in long-form video processing"""
    session = session_states.get(session_id)
    if not session or not session.get("long_form"):
        raise HTTPException(status_code=404, detail="Long-form session not found.")

    if clip_index >= session["total_clips"]:
        raise HTTPException(status_code=400, detail="Invalid clip index.")

    try:
        # Create clip directory
        clip_dir = os.path.join(session["frames_dir"], f"clip_{clip_index}")
        os.makedirs(clip_dir, exist_ok=True)

        # Calculate time range for this clip
        start_time = clip_index * config.clip_duration
        end_time = min((clip_index + 1) * config.clip_duration, session["duration"])

        logger.info(
            f"[{session_id}] Extracting clip {clip_index}: {start_time}s - {end_time}s"
        )

        # Extract frames for this clip
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            session["video_path"],
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-vf",
            f"fps={config.framerate}",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",  # Changed pixel format
            os.path.join(clip_dir, "%03d.jpg"),
        ]

        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

        # Initialize inference state for this clip
        inference_state = predictor.init_state(clip_dir)

        # Check if we need to apply mask from previous clip
        prev_mask_data = extract_last_frame_mask(session_id, clip_index)

        session["clips"][clip_index] = {
            "inference_state": inference_state,
            "clip_dir": clip_dir,
            "start_time": start_time,
            "end_time": end_time,
            "results": {},
            "prev_mask_applied": prev_mask_data is not None,
        }

        # Encode frames
        frames = []
        for filename in sorted(os.listdir(clip_dir)):
            if filename.endswith(".jpg"):
                with open(os.path.join(clip_dir, filename), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    frames.append(encoded_string)

        return {
            "session_id": session_id,
            "clip_index": clip_index,
            "frames": frames,
            "start_time": start_time,
            "end_time": end_time,
            "prev_mask_applied": prev_mask_data is not None,
        }

    except Exception as e:
        logger.error(f"Error creating clip session {clip_index} for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_new_points/")
async def add_new_points(data: ClickData):
    session = session_states.get(data.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    logger.debug(f"Point Format: {data}")

    if session.get("long_form"):
        raise HTTPException(
            status_code=400,
            detail="Use /add_points_clip/ for long-form video sessions.",
        )

    inference_state = session["inference_state"]
    frames_dir = session["frames_dir"]

    if data.resetState:
        predictor.reset_state_for_objectId(inference_state, data.objectId)
        logger.info(f"[{data.sessionId}] Reset state for object ID: {data.objectId}")

    with torch.inference_mode():
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=data.frameIndex,
            obj_id=data.objectId,
            points=np.array(data.points, dtype=np.float32),
            labels=np.array(data.labels, dtype=np.int32),
        )

    logger.info(f"objectId: {out_obj_ids}")

    rleMaskList = []
    preview_image = None

    # Generate preview image with mask overlay
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if frame_files and data.frameIndex < len(frame_files):
        frame_filename = frame_files[data.frameIndex]
        input_image_path = os.path.join(frames_dir, frame_filename)
        input_image = cv2.imread(input_image_path)
        overlay = input_image.copy()
        color_used = 0

        for idx, objId in enumerate(out_obj_ids):
            uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
            rleMaskList.append({"objectId": objId, "rleMask": uncompressed_rle})

            # Save mask metadata and images
            color = config.colors[color_used % len(config.colors)]
            save_mask_metadata(data.sessionId, data.frameIndex, objId, color)

            # Create preview mask
            color_used += 1
            combined_mask = None
            for rle_dict in uncompressed_rle:
                single_mask = rle_to_mask(rle_dict)
                single_mask_array = np.array(single_mask).reshape(input_image.shape[:2])
                if combined_mask is None:
                    combined_mask = single_mask_array
                else:
                    combined_mask = np.logical_or(combined_mask, single_mask_array)

            if combined_mask is not None:
                object_mask_bool = combined_mask.astype(bool)
                overlay[object_mask_bool] = color

                # Add border
                kernel = np.ones(
                    (config.border_thickness, config.border_thickness), np.uint8
                )
                dilated = cv2.dilate(
                    combined_mask.astype(np.uint8), kernel, iterations=1
                )
                eroded = cv2.erode(combined_mask.astype(np.uint8), kernel, iterations=1)
                border = dilated - eroded
                border_mask = border.astype(bool)
                overlay[border_mask] = (255, 255, 255)  # White border

        # Save mask images
        save_mask_images(data.sessionId, data.frameIndex, rleMaskList)

        # Blend with original image
        blended_image = cv2.addWeighted(
            overlay, config.mask_opacity, input_image, 1 - config.mask_opacity, 0
        )

        # Convert to base64
        _, buffer = cv2.imencode(".jpg", blended_image)
        preview_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "addPoints": {
            "frameIndex": frame_idx,
            "rleMaskList": rleMaskList,
            "previewImage": preview_image,
        }
    }


@app.post("/add_points_clip/")
async def add_points_clip(data: LongFormSegmentData):
    """Add points for long-form video clip segmentation"""
    session = session_states.get(data.sessionId)
    if not session or not session.get("long_form"):
        raise HTTPException(status_code=404, detail="Long-form session not found.")

    if data.clipIndex not in session["clips"]:
        raise HTTPException(
            status_code=400,
            detail="Clip not initialized. Call /create_clip_session/ first.",
        )

    clip_data = session["clips"][data.clipIndex]
    inference_state = clip_data["inference_state"]
    clip_dir = clip_data["clip_dir"]

    try:
        with torch.inference_mode():
            frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,  # Always start from first frame of clip
                obj_id=data.objectId,
                points=np.array(data.points, dtype=np.float32),
                labels=np.array(data.labels, dtype=np.int32),
            )

        rleMaskList = []
        for idx, objId in enumerate(out_obj_ids):
            uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
            rleMaskList.append({"objectId": objId, "rleMask": uncompressed_rle})

        return {
            "sessionId": data.sessionId,
            "clipIndex": data.clipIndex,
            "frameIndex": frame_idx,
            "rleMaskList": rleMaskList,
        }

    except Exception as e:
        logger.error(f"Error adding points to clip {data.clipIndex}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/propagate_clip/")
async def propagate_clip(session_id: str, clip_index: int):
    """Propagate masks for a specific clip"""
    session = session_states.get(session_id)
    if not session or not session.get("long_form"):
        raise HTTPException(status_code=404, detail="Long-form session not found.")

    if clip_index not in session["clips"]:
        raise HTTPException(status_code=400, detail="Clip not initialized.")

    clip_data = session["clips"][clip_index]
    inference_state = clip_data["inference_state"]

    try:
        logger.info(f"[{session_id}] Propagating masks for clip {clip_index}...")

        clip_data["results"] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            rleMaskList = []
            for idx, objId in enumerate(out_obj_ids):
                uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
                rleMaskList.append({"objectId": objId, "mask": uncompressed_rle})
            clip_data["results"][out_frame_idx] = rleMaskList

        # Clear GPU cache and reset state for next clip
        predictor.reset_state(inference_state)
        clear_gpu_cache()

        logger.info(
            f"[{session_id}] Clip {clip_index} propagation complete. {len(clip_data['results'])} frames processed."
        )

        return {
            "sessionId": session_id,
            "clipIndex": clip_index,
            "frameCount": len(clip_data["results"]),
            "message": f"Clip {clip_index} propagation complete.",
        }

    except Exception as e:
        logger.error(f"Error propagating clip {clip_index}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/propagate_in_video")
async def propagate_and_save_masks(data: PropagateData):
    sessionId = data.sessionId
    session = session_states.get(sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    if session.get("long_form"):
        raise HTTPException(
            status_code=400, detail="Use /propagate_clip/ for long-form video sessions."
        )

    inference_state = session["inference_state"]
    logger.info(f"[{sessionId}] Starting mask propagation...")

    session["results"] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        rleMaskList = []
        for idx, objId in enumerate(out_obj_ids):
            uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
            rleMaskList.append({"objectId": objId, "mask": uncompressed_rle})

            # Save mask metadata and images
            color = config.colors[objId % len(config.colors)]
            save_mask_metadata(sessionId, out_frame_idx, objId, color)

        session["results"][out_frame_idx] = rleMaskList

        # Save mask images for this frame
        save_mask_images(sessionId, out_frame_idx, rleMaskList)

    logger.info(
        f"[{sessionId}] Propagation complete. {len(session['results'])} frames processed."
    )
    predictor.reset_state(inference_state)
    clear_gpu_cache()

    return {
        "sessionId": sessionId,
        "message": "Propagation complete. Masks are generated and saved in session.",
        "frameCount": len(session["results"]),
    }


@app.post("/generate_longform_video")
async def generate_longform_video(session_id: str):
    """Generate final video from all processed clips"""
    session = session_states.get(session_id)
    if not session or not session.get("long_form"):
        raise HTTPException(status_code=404, detail="Long-form session not found.")

    if not config.save_videos:
        raise HTTPException(
            status_code=400, detail="Video generation is disabled in configuration."
        )

    try:
        logger.info(
            f"[{session_id}] Generating long-form video from {len(session['clips'])} clips..."
        )

        frames_dir = session["frames_dir"]
        output_dir = os.path.join(frames_dir, "final_output")
        os.makedirs(output_dir, exist_ok=True)

        frame_counter = 0

        # Process each clip in order
        for clip_index in sorted(session["clips"].keys()):
            clip_data = session["clips"][clip_index]
            clip_dir = clip_data["clip_dir"]
            clip_results = clip_data.get("results", {})

            if not clip_results:
                logger.warning(
                    f"[{session_id}] No results found for clip {clip_index}, skipping..."
                )
                continue

            frame_files = sorted(
                [f for f in os.listdir(clip_dir) if f.endswith(".jpg")]
            )

            for frame_filename in frame_files:
                input_image_path = os.path.join(clip_dir, frame_filename)
                try:
                    frame_idx_from_file = int(os.path.splitext(frame_filename)[0]) - 1
                except ValueError:
                    continue

                output_image_path = os.path.join(output_dir, f"{frame_counter:06d}.png")

                input_image = cv2.imread(input_image_path)
                if input_image is None:
                    continue

                overlay = input_image.copy()

                # Apply masks from results
                frame_results = clip_results.get(frame_idx_from_file)
                if frame_results:
                    for result in frame_results:
                        object_id = result.get("objectId", 0)
                        color = config.colors[object_id % len(config.colors)]
                        rle_list = result["mask"]

                        # Create combined mask
                        combined_mask = None
                        for rle_dict in rle_list:
                            single_mask = rle_to_mask(rle_dict)
                            single_mask_array = np.array(single_mask).reshape(
                                input_image.shape[:2]
                            )
                            if combined_mask is None:
                                combined_mask = single_mask_array
                            else:
                                combined_mask = np.logical_or(
                                    combined_mask, single_mask_array
                                )

                        if combined_mask is not None:
                            object_mask_bool = combined_mask.astype(bool)
                            overlay[object_mask_bool] = color

                            # Add border
                            kernel = np.ones(
                                (config.border_thickness, config.border_thickness),
                                np.uint8,
                            )
                            dilated = cv2.dilate(
                                combined_mask.astype(np.uint8), kernel, iterations=2
                            )
                            eroded = cv2.erode(
                                combined_mask.astype(np.uint8), kernel, iterations=1
                            )
                            border = dilated - eroded
                            border_mask = border.astype(bool)
                            overlay[border_mask] = (255, 255, 255)  # White border

                blended_image = cv2.addWeighted(
                    overlay,
                    config.mask_opacity,
                    input_image,
                    1 - config.mask_opacity,
                    0,
                )

                # Upscale to original resolution
                upsampled_blended_image = cv2.resize(
                    blended_image,
                    (session["original_width"], session["original_height"]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(output_image_path, upsampled_blended_image)
                frame_counter += 1

        logger.info(
            f"[{session_id}] All {frame_counter} frames processed. Creating final video..."
        )

        output_video_path = os.path.join(frames_dir, "longform_output.webm")
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(config.framerate),
            # Use os.path.join for the input image sequence path
            "-i",
            os.path.join(output_dir, "%06d.png"),
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",  # Changed pixel format
            "-b:v",
            "2M",
            output_video_path,
        ]

        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        shutil.rmtree(output_dir)

        logger.info(f"[{session_id}] Long-form video generation complete.")

        return FileResponse(
            output_video_path,
            media_type="video/webm",
            filename=f"longform_output_{session_id}.webm",
        )

    except Exception as e:
        logger.error(f"Error generating long-form video for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_video")
async def generate_video(data: GenerateData):
    session = session_states.get(data.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    if session.get("long_form"):
        raise HTTPException(
            status_code=400,
            detail="Use /generate_longform_video for long-form video sessions.",
        )

    if not config.save_videos:
        raise HTTPException(
            status_code=400, detail="Video generation is disabled in configuration."
        )

    results = session.get("results")
    if not results:
        raise HTTPException(
            status_code=400,
            detail="Masks have not been generated. Call /propagate_in_video first.",
        )

    frames_dir = session["frames_dir"]
    output_dir = os.path.join(frames_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"[{data.sessionId}] Generating video from masks...")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not frame_files:
        raise HTTPException(
            status_code=404, detail="No source frames found in session directory."
        )

    original_width = session.get("original_width")
    original_height = session.get("original_height")
    if not original_width or not original_height:
        raise HTTPException(
            status_code=500, detail="Original video dimensions not found in session."
        )

    for frame_filename in frame_files:
        input_image_path = os.path.join(frames_dir, frame_filename)
        try:
            frame_idx_from_file = int(os.path.splitext(frame_filename)[0])
        except ValueError:
            continue

        output_image_path = os.path.join(output_dir, f"{frame_idx_from_file:03d}.png")

        input_image = cv2.imread(input_image_path)
        overlay = input_image.copy()

        model_frame_idx = frame_idx_from_file - 1
        frame_results = results.get(model_frame_idx)

        if frame_results:
            for result in frame_results:
                object_id = result.get("objectId", 0)
                color = config.colors[object_id % len(config.colors)]
                rle_list = result["mask"]

                # Create combined mask
                combined_mask = None
                for rle_dict in rle_list:
                    single_mask = rle_to_mask(rle_dict)
                    single_mask_array = np.array(single_mask).reshape(
                        input_image.shape[:2]
                    )
                    if combined_mask is None:
                        combined_mask = single_mask_array
                    else:
                        combined_mask = np.logical_or(combined_mask, single_mask_array)

                if combined_mask is not None:
                    object_mask_bool = combined_mask.astype(bool)
                    overlay[object_mask_bool] = color

                    # Add border
                    kernel = np.ones(
                        (config.border_thickness, config.border_thickness), np.uint8
                    )
                    dilated = cv2.dilate(
                        combined_mask.astype(np.uint8), kernel, iterations=2
                    )
                    eroded = cv2.erode(
                        combined_mask.astype(np.uint8), kernel, iterations=1
                    )
                    border = dilated - eroded
                    border_mask = border.astype(bool)
                    overlay[border_mask] = (255, 255, 255)  # White border

        blended_image = cv2.addWeighted(
            overlay, config.mask_opacity, input_image, 1 - config.mask_opacity, 0
        )

        upsampled_blended_image = cv2.resize(
            blended_image,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.imwrite(output_image_path, upsampled_blended_image)

    logger.info(f"[{data.sessionId}] All frames masked. Creating final video file...")

    output_video_path = os.path.join(frames_dir, "output.webm")
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(config.framerate),
        # Use os.path.join for the input image sequence path
        "-i",
        os.path.join(output_dir, "%03d.png"),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuv420p",  # Changed pixel format
        "-b:v",
        "2M",
        output_video_path,
    ]

    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
    shutil.rmtree(output_dir)

    logger.info(f"[{data.sessionId}] Video generation complete. Sending file.")

    return FileResponse(
        output_video_path, media_type="video/webm", filename="output.webm"
    )


@app.get("/download_masks/{session_id}")
async def download_masks(session_id: str):
    """Download all masks as a ZIP file"""
    session = session_states.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    frames_dir = session["frames_dir"]
    masks_dir = os.path.join(frames_dir, "masks")
    metadata_dir = os.path.join(frames_dir, "masks_metadata")

    if not os.path.exists(masks_dir) and not os.path.exists(metadata_dir):
        raise HTTPException(status_code=404, detail="No masks found for this session.")

    # Create ZIP file
    zip_path = os.path.join(frames_dir, f"masks_{session_id}.zip")

    try:
        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add mask files
            if os.path.exists(masks_dir):
                for root, dirs, files in os.walk(masks_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # os.path.relpath will correctly handle paths on Windows
                        arcname = os.path.relpath(file_path, frames_dir)
                        zipf.write(file_path, arcname)

            # Add metadata files
            if os.path.exists(metadata_dir):
                for root, dirs, files in os.walk(metadata_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # os.path.relpath will correctly handle paths on Windows
                        arcname = os.path.relpath(file_path, frames_dir)
                        zipf.write(file_path, arcname)

        return FileResponse(
            zip_path, media_type="application/zip", filename=f"masks_{session_id}.zip"
        )

    except Exception as e:
        logger.error(f"Error creating masks ZIP for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session_info/{session_id}")
async def get_session_info(session_id: str):
    """Get detailed information about a session"""
    session = session_states.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    info = {
        "session_id": session_id,
        "long_form": session.get("long_form", False),
        "original_width": session.get("original_width"),
        "original_height": session.get("original_height"),
    }

    if session.get("long_form"):
        info.update(
            {
                "total_clips": session.get("total_clips", 0),
                "duration": session.get("duration", 0),
                "fps": session.get("fps", 0),
                "clip_duration": config.clip_duration,
                "clips_processed": len(session.get("clips", {})),
                "clips_info": [],
            }
        )

        for clip_index, clip_data in session.get("clips", {}).items():
            clip_info = {
                "clip_index": clip_index,
                "start_time": clip_data.get("start_time", 0),
                "end_time": clip_data.get("end_time", 0),
                "frames_processed": len(clip_data.get("results", {})),
                "prev_mask_applied": clip_data.get("prev_mask_applied", False),
            }
            info["clips_info"].append(clip_info)
    else:
        info.update(
            {
                "frames_processed": len(session.get("results", {})),
                "masks_saved": os.path.exists(
                    os.path.join(session["frames_dir"], "masks")
                ),
                "metadata_saved": os.path.exists(
                    os.path.join(session["frames_dir"], "masks_metadata")
                ),
            }
        )

    return info


@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: str):
    session = session_states.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    frames_dir = session.get("frames_dir")
    if frames_dir and os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)  # This works on Windows

    # Clear GPU cache
    clear_gpu_cache()

    logger.info(f"[{session_id}] Session deleted and files cleaned up.")
    return {"message": "Session deleted successfully."}


@app.delete("/cleanup_old_sessions")
async def cleanup_old_sessions(max_age_hours: int = 24):
    """Clean up old sessions based on file modification time"""
    import time

    cleaned_sessions = []
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for session_id, session in list(session_states.items()):
        frames_dir = session.get("frames_dir")
        if frames_dir and os.path.exists(frames_dir):
            dir_mtime = os.path.getmtime(frames_dir)
            if current_time - dir_mtime > max_age_seconds:
                try:
                    shutil.rmtree(frames_dir)  # This works on Windows
                    session_states.pop(session_id, None)
                    cleaned_sessions.append(session_id)
                    logger.info(f"Cleaned up old session: {session_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up session {session_id}: {e}")

    if cleaned_sessions:
        clear_gpu_cache()

    return {
        "message": f"Cleaned up {len(cleaned_sessions)} old sessions",
        "cleaned_sessions": cleaned_sessions,
        "remaining_sessions": len(session_states),
    }


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn

    # For Windows, make sure you have uvicorn installed: pip install "uvicorn[standard]"
    # Host '0.0.0.0' allows access from other machines on the network, '127.0.0.1' for local only.
    uvicorn.run(app, host="0.0.0.0", port=8000)
