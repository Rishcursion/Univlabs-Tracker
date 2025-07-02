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
obj_color = {}
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
    """POST endpoint for instantiating session with uploaded file.

    POST endpoint that handles creating sessions and storing the video uploaded by the user
    using postprocessing and ffmpeg, postprocessing includes downsampling the given frames
    by a pre-defined factor

    Args:
        video_file: The video to be used for inference
        start_time: The timestamp of the video from where inference should start
        end_time: The timestamp of the video until which inference should run.
        downsample_factor: The factor by which the video should be downsampled based on the
        original aspect ratio and resolution. i.e. 1280x720p downsampled by a factor of 2,
        would be stored as 480x360 frames.

    Raises:
        HTTPException: If FFmpeg fails to extract frames for the video or the ffmpeg command is not found.
        HTTPException: Unexpected error due to compromised frontend or other untested bugs. e.g. malformed data/unprocessable entity etc..
    """
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
    """POST endpoint for handling new points added by the user.

    POST endpoint where new points added by the user are utilized to generate a static mask for the
    current frame, so that the user will have an idea of how the mask will look like, i.e. real-time
    feedback. This endpoint also handles adding the points to the inference state of the model for
    mask propogation at a later stage

    Args:
        data: a JSON object, containing meta-data related to the points added by the user such as
        objectId, x-coordinate, y-coordinate, video timestamp etc.., refer to pydantic object
        ClickData, for the exact structure.

    Raises:
        HTTPException: If points are added but there are no related sessions active on the backend,
        an exception is raised.
    """
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
        start_col = 0
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
            color = colors[start_col]
            start_col += 1
            obj_color[objId] = color
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
    """POST endpoint for initializing MedSAM2 inference based on given mask prompts.

    POST endpoint that starts the propogation process using MedSAM2, only after
    annotations are available, results in masks being generated for the entire video.

    Args:
        data: PropagateData object, which tells the model from which timestamp of which session,
        inference needs to be done on.

    Raises:
        HTTPException: This exception is raised if session doesnt exist for the given data.
        OutOfMemoryError: This happens if there is not enough VRAM to process the entire video,
        resulting in inference being incomplete and then terminated.

    """
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
    """POST: Generates the final video based on the propogated masks.

    POST endpoint that generates the final preview video in webm format by overlaying
    the generated masks and the original frames together, assigning unique colors to
    each unique object, currently this is the biggest bottleneck and can be optimized
    in the future if computing time is a big concern.

    Args:
        data: Contains session id information.

    Raises:
        HTTPException: Raises if SessionID in the given payload is invalid.
        HTTPException: Raises if this endpoint is called before /propogate_in_video
        HTTPException: Raises if original extracted frames is not present in the specified path.
        HTTPException: Raises if video in the original resolution is not found in the specified path.
        HTTPException: Raises if either ffmpeg or cv2 experiences errors when processing the frames
    """
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

    # Pre-define colors as numpy arrays for faster access
    colors = np.array(
        [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [255, 192, 203],  # Pink
            [128, 255, 0],  # Lime
            [255, 165, 0],  # Dark Orange
            [0, 128, 255],  # Sky Blue
            [255, 69, 0],  # Red Orange
            [50, 205, 50],  # Lime Green
            [255, 20, 147],  # Deep Pink
            [0, 191, 255],  # Deep Sky Blue
            [255, 215, 0],  # Gold
            [138, 43, 226],  # Blue Violet
            [220, 20, 60],  # Crimson
            [32, 178, 170],  # Light Sea Green
        ],
        dtype=np.uint8,
    )

    # Pre-compute morphological kernels (optimized shapes)
    border_kernel = np.ones((3, 3), np.uint8)
    border_kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Option: Skip borders entirely for speed (uncomment next line)
    # SKIP_BORDERS = True  # 3-5x faster, but no white borders around masks
    SKIP_BORDERS = True

    # Process frames sequentially to ensure all files are created
    def process_single_frame(frame_filename, output_frame_idx):
        """Optimized single frame processing"""

        input_image_path = os.path.join(frames_dir, frame_filename)
        output_image_path = os.path.join(output_dir, f"{output_frame_idx:03d}.png")

        # Load image once
        input_image = cv2.imread(input_image_path)
        if input_image is None:
            logger.error(f"Could not load image: {input_image_path}")
            return False

        frame_height, frame_width = input_image.shape[:2]

        # Create overlay as a copy
        overlay = input_image.copy()

        # The model frame index corresponds to the propagation results
        model_frame_idx = (
            output_frame_idx - 1
        )  # Convert from 1-based to 0-based indexing

        frame_results = results.get(model_frame_idx)
        if frame_results:
            logger.debug(
                f"Processing frame {output_frame_idx} with {len(frame_results)} objects"
            )
            # Pre-allocate combined mask array
            all_masks = []
            object_colors = []

            for result in frame_results:
                obj_id = result["objectId"]
                # Get color for this object ID
                if obj_id in obj_color:
                    color = obj_color[obj_id]
                else:
                    # Assign a default color if not found
                    color_idx = obj_id % len(colors)
                    color = tuple(colors[color_idx].tolist())
                    obj_color[obj_id] = color

                # Process RLE masks more efficiently
                rle_list = result["mask"]
                combined_mask = None

                for rle_dict in rle_list:
                    single_mask = rle_to_mask(rle_dict)
                    single_mask_array = np.array(single_mask, dtype=np.uint8).reshape(
                        frame_height, frame_width
                    )

                    if combined_mask is None:
                        combined_mask = single_mask_array
                    else:
                        combined_mask = np.logical_or(
                            combined_mask, single_mask_array
                        ).astype(np.uint8)

                if combined_mask is not None:
                    all_masks.append(combined_mask)
                    object_colors.append(color)

            # Apply all masks in vectorized operations
            if all_masks:
                # FASTEST OPTION: Skip borders entirely (3-5x speed improvement)
                if SKIP_BORDERS:
                    for mask, color in zip(all_masks, object_colors):
                        mask_bool = mask.astype(bool)
                        overlay[mask_bool] = color

                # FAST OPTION: Approximate borders using contours (2-3x speed improvement)
                elif len(all_masks) > 2:
                    for mask, color in zip(all_masks, object_colors):
                        mask_bool = mask.astype(bool)
                        overlay[mask_bool] = color

                        # Fast border approximation using contours
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(
                            overlay, contours, -1, (255, 255, 255), thickness=2
                        )

                # MEDIUM SPEED: Batch morphological operations
                elif len(all_masks) > 1:
                    # Batch dilate and erode operations
                    dilated_masks = []
                    eroded_masks = []

                    for mask in all_masks:
                        dilated_masks.append(
                            cv2.dilate(mask, border_kernel, iterations=2)
                        )
                        eroded_masks.append(
                            cv2.erode(mask, border_kernel, iterations=1)
                        )

                    # Apply all masks and borders
                    for mask, color, dilated, eroded in zip(
                        all_masks, object_colors, dilated_masks, eroded_masks
                    ):
                        mask_bool = mask.astype(bool)
                        overlay[mask_bool] = color

                        border = dilated - eroded
                        border_mask = border.astype(bool)
                        overlay[border_mask] = [255, 255, 255]

                # SLOW BUT PRECISE: Single-pass morphological gradient
                else:
                    for mask, color in zip(all_masks, object_colors):
                        mask_bool = mask.astype(bool)
                        overlay[mask_bool] = color

                        # Single-pass border detection using morphological gradient
                        border = cv2.morphologyEx(
                            mask, cv2.MORPH_GRADIENT, border_kernel_large
                        )
                        border_mask = border.astype(bool)
                        overlay[border_mask] = [255, 255, 255]

        # Blend images
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0, overlay)

        # Resize only once at the end
        if (frame_width != original_width) or (frame_height != original_height):
            overlay = cv2.resize(
                overlay,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Write output
        success = cv2.imwrite(output_image_path, overlay)
        if not success:
            logger.error(f"Failed to write image: {output_image_path}")
            return False

        logger.debug(f"Successfully wrote frame: {output_image_path}")
        return True

    # Process all frames sequentially to ensure consistency
    successful_frames = 0
    for i, frame_filename in enumerate(frame_files):
        output_frame_idx = i + 1  # FFmpeg expects 1-based indexing for %03d format

        success = process_single_frame(frame_filename, output_frame_idx)
        if success:
            successful_frames += 1

        # Log progress every 10 frames
        if (i + 1) % 10 == 0:
            logger.info(
                f"[{data.sessionId}] Processed {i + 1}/{len(frame_files)} frames"
            )

    logger.info(
        f"[{data.sessionId}] Successfully processed {successful_frames}/{len(frame_files)} frames"
    )

    # Verify that output files exist before calling FFmpeg
    output_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    if not output_files:
        raise HTTPException(
            status_code=500,
            detail="No output frames were generated. Check mask propagation results.",
        )

    logger.info(
        f"[{data.sessionId}] Found {len(output_files)} output frames: {output_files[:5]}..."
    )

    logger.info(f"[{data.sessionId}] All frames masked. Creating final video file...")

    # Use a fallback FFmpeg command if NVENC is not available
    output_video_path = os.path.join(frames_dir, "output.webm")

    # Try NVENC first, fall back to software encoding if it fails
    ffmpeg_commands = [
        # NVENC command (faster if available)
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "24",
            "-i",
            f"{output_dir}/%03d.png",
            "-c:v",
            "vp9_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "2M",
            "-rc",
            "vbr",
            "-cq",
            "28",
            "-preset",
            "fast",
            output_video_path,
        ],
        # Software fallback command
        [
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
            "-crf",
            "28",
            "-preset",
            "fast",
            output_video_path,
        ],
    ]

    video_created = False
    for i, ffmpeg_command in enumerate(ffmpeg_commands):
        try:
            result = subprocess.run(
                ffmpeg_command, check=True, capture_output=True, text=True
            )
            logger.info(
                f"[{data.sessionId}] Video created successfully using {'NVENC' if i == 0 else 'software'} encoding"
            )
            video_created = True
            break
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg command {i+1} failed: {e.stderr}")
            if i == len(ffmpeg_commands) - 1:  # Last command failed
                logger.error(f"All FFmpeg commands failed. Last error: {e.stderr}")
                raise HTTPException(
                    status_code=500, detail=f"Video encoding failed: {e.stderr}"
                )

    if not video_created:
        raise HTTPException(
            status_code=500, detail="Failed to create video with any encoding method"
        )

    # Cleanup
    shutil.rmtree(output_dir)

    logger.info(f"[{data.sessionId}] Video generation complete. Sending file.")
    return FileResponse(
        output_video_path, media_type="video/webm", filename="output.webm"
    )


@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: str):
    """Used for invalidating a specified session.

    Invalidates a specified session and frees up resources used by the session to be deleted.

    Args:
        session_id: string representing the session to delete.

    Raises:
        HTTPException: Raises when a session_id that does not exist is requested to be deleted.
    """
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
