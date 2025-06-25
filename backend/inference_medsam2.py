import base64
import logging
import os
import shutil
import subprocess
import uuid
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Correctly import the MedSAM2 video predictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.amg import mask_to_rle_pytorch, rle_to_mask

# --- Basic Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Model and Session Management ---
# Dictionary to hold session states
session_states = {}

try:
    # IMPORTANT: Update these paths to your MedSAM2 model and config file
    checkpoint = "./MedSAM2/checkpoints/MedSAM2_latest.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t512.yaml"

    # Using 'cuda' device as referenced in app.py for better performance
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")
    logger.info("Successfully loaded MedSAM2 model on CUDA device.")

except Exception as e:
    logger.error(f"Failed to load MedSAM2 model: {e}")
    logger.error(
        "Please ensure the checkpoint and model_cfg paths are correct and the model files exist."
    )
    predictor = None


# --- Pydantic Models for API Data Validation ---
class CreateSessionData(BaseModel):
    s3_link: str


class CreateSessionResponse(BaseModel):
    session_id: str
    frames: List[str]


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


# --- API Endpoints ---


@app.on_event("startup")
async def startup_event():
    if predictor is None:
        raise RuntimeError(
            "MedSAM2 predictor could not be initialized. Check model paths and dependencies."
        )
    os.makedirs("./tmp", exist_ok=True)
    logger.info("Application startup complete. Model loaded and tmp directory ensured.")


@app.post("/create_session_upload/", response_model=CreateSessionResponse)
async def create_session_upload(
    video_file: UploadFile = File(...),
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    downsample_factor: Optional[int] = None,
):
    session_id = str(uuid.uuid4())
    frames_dir = f"./tmp/{session_id}"
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

        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(
            f"[{session_id}] Original video resolution: {original_width}x{original_height}"
        )
        logger.info(f"[{session_id}] Video uploaded. Extracting frames...")

        ffmpeg_command = ["ffmpeg", "-i", video_path]

        if start_time:
            ffmpeg_command.extend(["-ss", start_time])
        if end_time:
            ffmpeg_command.extend(["-to", end_time])

        video_filters = ["fps=24"]
        if downsample_factor and downsample_factor > 1:
            video_filters.append(f"scale=iw/{downsample_factor}:ih/{downsample_factor}")
            logger.info(
                f"[{session_id}] Downsampling frames by a factor of {downsample_factor}."
            )

        ffmpeg_command.extend(["-vf", ",".join(video_filters)])
        ffmpeg_command.extend(
            [
                "-q:v",
                "2",
                "-pix_fmt",
                "yuvj444p",
                f"{frames_dir}/%03d.jpg",
            ]
        )

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
        }

        frames = []
        for filename in sorted(os.listdir(frames_dir)):
            if filename.endswith(".jpg"):
                with open(os.path.join(frames_dir, filename), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
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
            f"An unexpected error occurred in create_session_upload for session {session_id}: {e}"
        )
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_new_points/")
async def add_new_points(data: ClickData):
    session = session_states.get(data.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    logger.debug(f"Point Format : {data}")
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
    logger.info(f"objectId : {out_obj_ids}")
    rleMaskList = []
    preview_image = None

    # Generate preview image with mask overlay
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if frame_files and data.frameIndex < len(frame_files):
        frame_filename = frame_files[data.frameIndex]
        input_image_path = os.path.join(frames_dir, frame_filename)
        input_image = cv2.imread(input_image_path)
        overlay = input_image.copy()

        colors = [
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

        for idx, objId in enumerate(out_obj_ids):
            uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
            rleMaskList.append({"objectId": objId, "rleMask": uncompressed_rle})

            # Create preview mask
            color = colors[objId % len(colors)]
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

                # Add filled mask
                overlay[object_mask_bool] = color

                # Add border
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(
                    combined_mask.astype(np.uint8), kernel, iterations=1
                )
                eroded = cv2.erode(combined_mask.astype(np.uint8), kernel, iterations=1)
                border = dilated - eroded
                border_mask = border.astype(bool)
                overlay[border_mask] = (255, 255, 255)  # White border

        # Blend with original image
        alpha = 0.6
        blended_image = cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0)

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


@app.post("/propagate_in_video")
async def propagate_and_save_masks(data: PropagateData):
    sessionId = data.sessionId
    session = session_states.get(sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    inference_state = session["inference_state"]
    logger.info(f"[{sessionId}] Starting mask propagation...")

    session["results"] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        rleMaskList = []
        for idx, objId in enumerate(out_obj_ids):
            # MODIFIED: Store the entire RLE list to correctly handle multi-part masks.
            uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
            rleMaskList.append({"objectId": objId, "mask": uncompressed_rle})
        session["results"][out_frame_idx] = rleMaskList

    logger.info(
        f"[{sessionId}] Propagation complete. {len(session['results'])} frames processed."
    )
    predictor.reset_state(inference_state)
    return {
        "sessionId": sessionId,
        "message": "Propagation complete. Masks are generated and saved in session.",
        "frameCount": len(session["results"]),
    }


@app.post("/generate_video")
async def generate_video(data: GenerateData):
    session = session_states.get(data.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

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

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

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
                color = colors[object_id % len(colors)]

                # FIXED: Handle the RLE list properly
                rle_list = result["mask"]

                # Create a combined mask from all RLE parts
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

                    # Add filled mask
                    overlay[object_mask_bool] = color

                    # Add border to masks
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate(
                        combined_mask.astype(np.uint8), kernel, iterations=2
                    )
                    eroded = cv2.erode(
                        combined_mask.astype(np.uint8), kernel, iterations=1
                    )
                    border = dilated - eroded
                    border_mask = border.astype(bool)
                    overlay[border_mask] = (255, 255, 255)  # White border

        alpha = 0.5
        blended_image = cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0)

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
        "24",
        "-i",
        f"{output_dir}/%03d.png",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuv420p",
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


@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: str):
    session = session_states.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    frames_dir = session.get("frames_dir")
    if frames_dir and os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    logger.info(f"[{session_id}] Session deleted and files cleaned up.")
    return {"message": "Session deleted successfully."}


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
