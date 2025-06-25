import base64
import logging
import os
import shutil
import subprocess
import uuid
from typing import List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

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

# Load the model checkpoint and configuration
# Ensure these paths are correct relative to where you run the server
try:
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cpu")
except Exception as e:
    logger.error(f"Failed to load SAM model: {e}")
    # Exit or handle gracefully if the model can't be loaded
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
            "SAM predictor could not be initialized. Check model paths and dependencies."
        )
    # Reset model state
    # Create a temporary directory for processing files if it doesn't exist
    os.makedirs("./tmp", exist_ok=True)
    logger.info("Application startup complete. Model loaded and tmp directory ensured.")


# Endpoint for creating a session from a direct video file upload
@app.post("/create_session_upload/", response_model=CreateSessionResponse)
async def create_session_upload(video_file: UploadFile = File(...)):
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

        logger.info(f"[{session_id}] Video uploaded. Extracting downsampled frames...")
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            "fps=24",  # <- Downsample + framerate
            "-q:v",
            "2",
            "-pix_fmt",
            "yuvj444p",
            f"{frames_dir}/%03d.jpg",
        ]
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        os.remove(video_path)
        logger.info(f"[{session_id}] Frame extraction complete.")

        inference_state = predictor.init_state(frames_dir)

        # Initialize the session state, including an empty results dict
        session_states[session_id] = {
            "inference_state": inference_state,
            "frames_dir": frames_dir,
            "results": {},  # Initialize results here
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


# Endpoint for adding points to an object in a frame
@app.post("/add_new_points/")
async def add_new_points(data: ClickData):
    session = session_states.get(data.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    logger.debug(f"Point Format : {data}")
    inference_state = session["inference_state"]

    if data.resetState:
        predictor.reset_state_for_objectId(inference_state, data.objectId)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=data.frameIndex,
            obj_id=data.objectId,
            points=np.array(data.points, dtype=np.float32),
            labels=np.array(data.labels, dtype=np.int32),
            clear_old_points=True,
            normalize_coords=False,
        )

    rleMaskList = []
    for idx, objId in enumerate(out_obj_ids):
        uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
        rleMaskList.append({"objectId": objId, "rleMask": uncompressed_rle[0]})

    return {"addPoints": {"frameIndex": frame_idx, "rleMaskList": rleMaskList}}


# REVISED: Synchronous endpoint to propagate masks through the entire video
@app.post("/propagate_in_video")
async def propagate_and_save_masks(data: PropagateData):
    sessionId = data.sessionId
    session = session_states.get(sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    inference_state = session["inference_state"]
    logger.info(f"[{sessionId}] Starting mask propagation...")

    # Clear any previous results and run the full propagation
    session["results"] = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        rleMaskList = []
        for idx, objId in enumerate(out_obj_ids):
            uncompressed_rle = mask_to_rle_pytorch(out_mask_logits[idx] > 0.0)
            rleMaskList.append({"objectId": objId, "mask": uncompressed_rle[0]})
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


# REVISED: Endpoint to generate the final video with a colored overlay
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

    # Define a list of BGR colors for different object masks
    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Process each frame
    for frame_filename in frame_files:
        input_image_path = os.path.join(frames_dir, frame_filename)

        try:
            frame_idx_from_file = int(os.path.splitext(frame_filename)[0])
        except ValueError:
            continue

        output_image_path = os.path.join(output_dir, f"{frame_idx_from_file:03d}.png")

        # Load the original image
        input_image = cv2.imread(input_image_path)
        # Create a copy to draw the overlay on
        overlay = input_image.copy()

        # Model index is 0-based, so subtract 1 from the 1-based file index
        model_frame_idx = frame_idx_from_file - 1
        frame_results = results.get(model_frame_idx)

        if frame_results:
            for result in frame_results:
                object_id = result.get("objectId", 0)
                color = colors[object_id % len(colors)]

                object_mask = rle_to_mask(result["mask"])
                # Ensure mask is a boolean array for indexing
                object_mask_bool = (
                    np.array(object_mask).reshape(input_image.shape[:2]).astype(bool)
                )

                # Apply color to the overlay where the mask is true
                overlay[object_mask_bool] = color

        # Blend the colored overlay with the original image
        alpha = 0.5  # Opacity of the overlay
        blended_image = cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0)

        # Save the resulting blended image
        cv2.imwrite(output_image_path, blended_image)

    logger.info(f"[{data.sessionId}] All frames masked. Creating final video file...")
    output_video_path = os.path.join(frames_dir, "output.webm")

    # Updated ffmpeg command for standard opaque video
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
        "yuv420p",  # Standard pixel format for web video
        "-b:v",
        "2M",  # Set a reasonable bitrate
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
