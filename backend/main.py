import base64
import json
import logging
import os
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sam2_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Directories
FRAME_DIR = "./test"
OUTPUT_DIR = "./visualization"
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build SAM2 model
logger.info("Initializing SAM2 model...")
model = build_sam2_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_s.yaml",
    ckpt_path="./checkpoints/sam2.1_hiera_small.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
logger.info(
    f"SAM2 model loaded on device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
)

# FastAPI app
server_start_time = time.time()
app = FastAPI(title="SAM2 Backend")


@app.get("/")
async def get():
    uptime = time.time() - server_start_time
    logger.info(f"Health check - Server uptime: {uptime:.2f} seconds")
    return {
        "status": "OK",
        "uptime": f"{uptime:.2f} seconds",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.websocket("/get_inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    prompt_info = {}
    received_frames = {}  # Changed to dict to store frame data by index

    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    logger.info(
                        f"Received payload type: {data.get('type', 'main_payload')}"
                    )

                    # Handle frame data (sent separately with type: 'frame')
                    if data.get("type") == "frame":
                        idx = data["index"]
                        img_data = data["data"]

                        # Save base64-encoded image
                        if img_data.startswith("data:image"):
                            logger.info(f"Processing frame {idx}")
                            img_bytes = base64.b64decode(img_data.split(",")[1])
                            img_path = os.path.join(FRAME_DIR, f"{idx:06d}.jpg")
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            received_frames[idx] = img_path
                            logger.info(f"Saved frame {idx} to {img_path}")

                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "status": "success",
                                        "message": f"Frame {idx} received and saved",
                                        "frame_count": len(received_frames),
                                    }
                                )
                            )

                    # Handle main payload (annotation data)
                    else:
                        # This is the main payload with video info and annotations
                        prompt_info = data
                        video_info = data.get("video", {})
                        fps = data.get("fps", 30)
                        current_time = data.get("currentTime", 0)
                        annotations = data.get("annotations", [])

                        logger.info(
                            f"Received main payload for video: {video_info.get('name', 'unknown')}"
                        )
                        logger.info(f"Video FPS: {fps}")
                        logger.info(f"Current time: {current_time}")
                        logger.info(f"Number of annotations: {len(annotations)}")

                        # Log annotation details
                        for i, ann in enumerate(annotations):
                            logger.info(
                                f"Annotation {i+1}: timestamp={ann.get('timestamp', 'N/A')}"
                            )

                        # Wait a moment for frames to arrive (since they're sent after the main payload)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "status": "info",
                                    "message": "Main payload received, waiting for frames...",
                                }
                            )
                        )

                        # Store the payload for when we have all frames
                        prompt_info["waiting_for_frames"] = True

                    # Check if we have both payload and frames for inference
                    if (
                        prompt_info.get("waiting_for_frames")
                        and len(received_frames) >= 30
                    ):  # Expecting 30 frames based on frontend

                        logger.info("All frames received, starting inference...")

                        try:
                            # Run SAM2 inference
                            masks = run_sam2_inference(prompt_info, received_frames)
                            logger.info(
                                f"Inference completed. Generated {len(masks)} masks"
                            )

                            # Save masks and send results
                            mask_paths = []
                            for i, mask in enumerate(masks):
                                out_path = os.path.join(OUTPUT_DIR, f"mask_{i:06d}.png")

                                # Handle different mask formats
                                if isinstance(mask, torch.Tensor):
                                    mask_np = mask.cpu().numpy()
                                else:
                                    mask_np = mask

                                # Ensure mask is in correct format
                                if mask_np.dtype != np.uint8:
                                    mask_np = (mask_np * 255).astype(np.uint8)

                                cv2.imwrite(out_path, mask_np)
                                mask_paths.append(out_path)
                                logger.info(
                                    f"Saved mask {i+1}/{len(masks)} to {out_path}"
                                )

                            video_info = prompt_info.get("video", {})
                            annotations = prompt_info.get("annotations", [])

                            result = {
                                "status": "success",
                                "message": "Inference complete. Masks saved.",
                                "mask_count": len(masks),
                                "mask_paths": mask_paths,
                                "video_name": video_info.get("name", "unknown"),
                                "annotation_count": len(annotations),
                                "frame_count": len(received_frames),
                            }

                            logger.info(
                                f"Inference pipeline completed successfully for {video_info.get('name', 'unknown')}"
                            )
                            await websocket.send_text(json.dumps(result))

                            # Reset state for next inference
                            prompt_info = {}
                            received_frames = {}

                        except Exception as e:
                            error_msg = f"Inference error: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            await websocket.send_text(
                                json.dumps({"status": "error", "message": error_msg})
                            )

                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON received: {str(e)}"
                    logger.error(error_msg)
                    await websocket.send_text(
                        json.dumps({"status": "error", "message": error_msg})
                    )

            elif "bytes" in msg:
                logger.warning("Received unsupported bytes data")
                await websocket.send_text(
                    json.dumps(
                        {"status": "error", "message": "Bytes data not supported"}
                    )
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_text(
                json.dumps({"status": "error", "message": f"Server error: {str(e)}"})
            )
        except:
            logger.error("Failed to send error message to client")


def run_sam2_inference(
    prompt_json: Dict[str, Any], received_frames: Dict[int, str]
) -> List[np.ndarray]:
    """
    Run SAM2 inference with the updated payload format and received frames
    """
    video_info = prompt_json.get("video", {})
    annotations = prompt_json.get("annotations", [])
    fps = prompt_json.get("fps", 30)

    logger.info(f"Processing video: {video_info.get('name', 'unknown')}")
    logger.info(f"Video FPS: {fps}")

    if not annotations:
        raise ValueError("No annotations provided for inference")

    if not received_frames:
        raise ValueError("No frames received for inference")

    points = []
    labels = []
    frames_with_annotations = set()

    # Process annotations - need to convert timestamp to frame index
    for ann in annotations:
        timestamp = ann.get("timestamp", 0)
        frame_idx = int(timestamp * fps)  # Convert timestamp to frame index

        # For now, assuming we're getting point annotations
        # You may need to adjust this based on your actual annotation format
        if "coordinates" in ann:
            coords = ann["coordinates"]  # [x, y] format
            points.append([coords[0], coords[1]])
            labels.append(1)  # Default to positive points
            frames_with_annotations.add(frame_idx)
            logger.info(
                f"Added point at ({coords[0]}, {coords[1]}) for frame {frame_idx} (timestamp: {timestamp})"
            )
        elif "x" in ann and "y" in ann:
            points.append([ann["x"], ann["y"]])
            labels.append(1)  # Default to positive points
            frames_with_annotations.add(frame_idx)
            logger.info(
                f"Added point at ({ann['x']}, {ann['y']}) for frame {frame_idx} (timestamp: {timestamp})"
            )

    if not points:
        raise ValueError("No valid points found in annotations")

    logger.info(
        f"Total points: {len(points)} (across {len(frames_with_annotations)} frames)"
    )

    # Load frames from received_frames dict
    frame_indices = sorted(received_frames.keys())
    frames = []

    for idx in frame_indices:
        frame_path = received_frames[idx]
        try:
            img = Image.open(frame_path).convert("RGB")
            frames.append(np.array(img))
            if len(frames) == 1:
                h, w = frames[0].shape[:2]
                logger.info(f"Frame dimensions: {w}x{h}")
        except Exception as e:
            logger.error(f"Error loading frame {frame_path}: {e}")
            continue

    if not frames:
        raise ValueError("No valid frames could be loaded")

    logger.info(f"Successfully loaded {len(frames)} frames for inference")

    # Initialize inference state with frames
    logger.info("Initializing SAM2 inference state with frames...")
    inference_state = model.init_state(FRAME_DIR)

    # Add prompts for the first frame (frame 0)
    frame_idx = 0
    obj_id = 1

    logger.info(f"Adding prompts to frame {frame_idx} with {len(points)} points")

    # Add new points to the specified frame
    _, out_obj_ids, out_mask_logits = model.add_new_points(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=np.array(points),
        labels=np.array(labels),
    )

    # Propagate the prompts to get masks for all frames
    logger.info("Propagating prompts through video...")
    masks = []

    video_segments = {}  # Store all video segments
    for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            "obj_ids": out_obj_ids,
            "mask_logits": out_mask_logits,
        }

    # Extract masks from video segments
    for frame_idx in sorted(video_segments.keys()):
        frame_data = video_segments[frame_idx]
        for i, out_obj_id in enumerate(frame_data["obj_ids"]):
            mask = frame_data["mask_logits"][i] > 0.0  # Convert logits to binary mask
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            masks.append(mask.squeeze())

    logger.info(f"Generated {len(masks)} masks from {len(video_segments)} frames")
    return masks
