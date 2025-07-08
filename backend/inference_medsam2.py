import base64
import logging
import os
import shutil
import uuid
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sam2.build_sam import build_sam2_video_predictor

# --- Basic Setup & Utility Functions ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
session_states = {}
obj_color = {}
CHUNK_SIZE = 180  # Process 300 frames at a time

try:
    checkpoint = "./MedSAM2/checkpoints/MedSAM2_latest.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t512.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")
    logger.info("Successfully loaded MedSAM2 model.")
except Exception as e:
    logger.error(f"Failed to load MedSAM2 model: {e}")
    predictor = None


def hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    if len(hex_color) != 6:
        return 0, 255, 0
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# --- Pydantic Models ---
class CreateSessionResponse(BaseModel):
    session_id: str
    message: str
    frame_count: int
    fps: float


class ClickData(BaseModel):
    sessionId: str
    frameIndex: int
    objectId: int
    labels: List[int]
    points: List[List[float]]
    color: Optional[str] = None


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    if predictor is None:
        raise RuntimeError("MedSAM2 predictor could not be initialized.")
    os.makedirs("./tmp", exist_ok=True)
    logger.info("Application startup complete.")


@app.post("/create_session_upload/", response_model=CreateSessionResponse)
async def create_session_upload(video_file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = f"./tmp/{session_id}"
    try:
        os.makedirs(session_dir, exist_ok=True)
        video_path = os.path.join(session_dir, "source_video.mp4")
        with open(video_path, "wb") as f:
            f.write(await video_file.read())
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        session_states[session_id] = {
            "video_path": video_path,
            "session_dir": session_dir,
            "original_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "original_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": frame_count,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "user_points": {},
        }
        cap.release()
        return CreateSessionResponse(
            session_id=session_id,
            message="Session created.",
            frame_count=frame_count,
            fps=session_states[session_id]["fps"],
        )
    except Exception as e:
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_frame/{session_id}/{frame_index}")
async def get_frame(session_id: str, frame_index: int = Path(..., ge=0)):
    session = session_states.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    cap = cv2.VideoCapture(session["video_path"])
    if frame_index >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.release()
        raise HTTPException(status_code=404, detail="Frame index out of bounds.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise HTTPException(status_code=404, detail="Could not read frame.")
    _, buffer = cv2.imencode(".jpg", frame)
    return {"frame": base64.b64encode(buffer).decode("utf-8")}


@app.post("/add_new_points/")
async def add_new_points(data: ClickData):
    session = session_states.get(data.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    frame_index_str = str(data.frameIndex)
    if frame_index_str not in session["user_points"]:
        session["user_points"][frame_index_str] = []
    session["user_points"][frame_index_str].append(
        {"objectId": data.objectId, "points": data.points, "labels": data.labels}
    )
    if data.color:
        obj_color[data.objectId] = hex_to_bgr(data.color)
    cap = cv2.VideoCapture(session["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, data.frameIndex)
    success, input_image = cap.read()
    cap.release()
    if not success:
        raise HTTPException(status_code=404, detail="Frame not found for preview.")
    temp_frames_dir = os.path.join(session["session_dir"], "temp_preview")
    os.makedirs(temp_frames_dir, exist_ok=True)
    cv2.imwrite(os.path.join(temp_frames_dir, "000.jpg"), input_image)
    preview_state = predictor.init_state(temp_frames_dir)
    current_frame_annotations = {}
    for p_data in session["user_points"][frame_index_str]:
        obj_id = p_data["objectId"]
        if obj_id not in current_frame_annotations:
            current_frame_annotations[obj_id] = {"points": [], "labels": []}
        current_frame_annotations[obj_id]["points"].extend(p_data["points"])
        current_frame_annotations[obj_id]["labels"].extend(p_data["labels"])
    all_out_obj_ids, all_out_mask_logits = [], None
    with torch.inference_mode():
        for obj_id, annotation_data in current_frame_annotations.items():
            _, all_out_obj_ids, all_out_mask_logits = predictor.add_new_points(
                inference_state=preview_state,
                frame_idx=0,
                obj_id=obj_id,
                points=np.array(annotation_data["points"], dtype=np.float32),
                labels=np.array(annotation_data["labels"], dtype=np.int32),
            )
    shutil.rmtree(temp_frames_dir)
    overlay = input_image.copy()
    H, W, _ = input_image.shape
    if all_out_mask_logits is not None:
        for idx, obj_id in enumerate(all_out_obj_ids):
            color = obj_color.get(obj_id, (0, 255, 0))
            mask = (all_out_mask_logits[idx] > 0.0).cpu().numpy()
            overlay[np.reshape(mask, (H, W))] = color
    blended_image = cv2.addWeighted(overlay, 0.6, input_image, 0.4, 0)
    _, buffer = cv2.imencode(".jpg", blended_image)
    return {"previewImage": base64.b64encode(buffer).decode("utf-8")}


# In backend/inference_medsam2.py


@app.post("/process_video/{session_id}")
async def process_video(session_id: str):
    session = session_states.get(session_id)
    if not session or not session.get("user_points"):
        raise HTTPException(status_code=404, detail="Session or user points not found.")

    # --- Setup ---
    video_path, session_dir = session["video_path"], session["session_dir"]
    frame_count, H, W = (
        session["frame_count"],
        session["original_height"],
        session["original_width"],
    )
    output_video_path = os.path.join(session_dir, "output.webm")

    # Define chunking parameters with overlap
    OVERLAP_SIZE = 30
    EFFECTIVE_CHUNK_SIZE = CHUNK_SIZE - OVERLAP_SIZE

    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out_writer = cv2.VideoWriter(output_video_path, fourcc, session["fps"], (W, H))

    last_chunk_hand_off_info = None
    total_frames_processed = 0

    # The main loop now steps by the effective (non-overlapping) chunk size
    for effective_start_frame in range(0, frame_count, EFFECTIVE_CHUNK_SIZE):
        # 1. Calculate the actual frame range for processing, including the overlap
        actual_start_frame = max(0, effective_start_frame - OVERLAP_SIZE)
        chunk_end_frame = min(effective_start_frame + EFFECTIVE_CHUNK_SIZE, frame_count)

        logger.info(
            f"--- Processing frames {actual_start_frame} to {chunk_end_frame-1} (writing from {effective_start_frame}) ---"
        )

        # 2. Extract frames for the current processing window (chunk + overlap)
        chunk_frames_dir = os.path.join(session_dir, f"chunk_{effective_start_frame}")
        os.makedirs(chunk_frames_dir, exist_ok=True)

        cap_extract = cv2.VideoCapture(video_path)
        cap_extract.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)
        frames_extracted_count = 0
        for i in range(actual_start_frame, chunk_end_frame):
            success, frame = cap_extract.read()
            if not success:
                break
            # Frame names are relative to the chunk's start (00000.jpg, 00001.jpg, etc.)
            cv2.imwrite(
                os.path.join(chunk_frames_dir, f"{i - actual_start_frame:05d}.jpg"),
                frame,
            )
            frames_extracted_count += 1
        cap_extract.release()

        if frames_extracted_count == 0:
            logger.error(
                f"No frames extracted for chunk starting at {actual_start_frame}. Stopping."
            )
            break

        # 3. Initialize predictor state on the chunk's frames
        inference_state = predictor.init_state(chunk_frames_dir)

        # 4. Conditional Prompting
        if effective_start_frame == 0:
            logger.info("Initializing first chunk with user point annotations.")
            for frame_idx_str, points_data_list in session["user_points"].items():
                relative_frame_idx = int(
                    frame_idx_str
                )  # No subtraction as it's the first chunk
                if 0 <= relative_frame_idx < frames_extracted_count:
                    for p_data in points_data_list:
                        predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=relative_frame_idx,
                            obj_id=p_data["objectId"],
                            points=np.array(p_data["points"], dtype=np.float32),
                            labels=np.array(p_data["labels"], dtype=np.int32),
                        )
        elif last_chunk_hand_off_info is not None:
            logger.info("Initializing chunk with centroids from previous chunk.")
            for obj_id, centroid in last_chunk_hand_off_info.items():
                # Apply the prompt to the correct relative frame index, which is the end of the overlap
                prompt_frame_idx = OVERLAP_SIZE - 1
                predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_idx,
                    obj_id=obj_id,
                    points=np.array([centroid], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )

        # 5. Propagate through the chunk
        with torch.inference_mode():
            for (
                relative_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(inference_state):
                absolute_frame_idx = actual_start_frame + relative_frame_idx
                if absolute_frame_idx >= chunk_end_frame:
                    break

                # IMPORTANT: Skip writing frames from the overlapping part to avoid duplicates
                if absolute_frame_idx < effective_start_frame:
                    continue

                total_frames_processed += 1

                frame_path = os.path.join(
                    chunk_frames_dir, f"{relative_frame_idx:05d}.jpg"
                )
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                overlay = frame.copy()

                # If it's the last frame of the processing window, save the hand-off info
                if absolute_frame_idx == chunk_end_frame - 1:
                    last_chunk_hand_off_info = {}

                for idx, obj_id in enumerate(out_obj_ids):
                    mask_logit = out_mask_logits[idx]
                    reshaped_mask = np.reshape((mask_logit > 0.0).cpu().numpy(), (H, W))
                    overlay[reshaped_mask] = obj_color.get(obj_id, (255, 0, 0))

                    if absolute_frame_idx == chunk_end_frame - 1:
                        coords = np.argwhere(reshaped_mask)
                        if len(coords) > 0:
                            centroid_y, centroid_x = coords.mean(axis=0)
                            last_chunk_hand_off_info[obj_id] = (centroid_x, centroid_y)

                blended_frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                out_writer.write(blended_frame)

        # 6. Cleanup for the current chunk
        logger.info(f"Resetting GPU state for chunk ending at {chunk_end_frame-1}")
        predictor.reset_state(inference_state)
        del inference_state
        torch.cuda.empty_cache()
        shutil.rmtree(chunk_frames_dir)

    out_writer.release()
    if not os.path.exists(output_video_path) or total_frames_processed == 0:
        raise HTTPException(
            status_code=500,
            detail="Output video could not be generated. Model failed to track the object.",
        )

    logger.info(f"[{session_id}] Video processing complete.")
    return FileResponse(
        output_video_path, media_type="video/webm", filename="output.webm"
    )


@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: str):
    session = session_states.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    session_dir = session.get("session_dir")
    if session_dir and os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    logger.info(f"[{session_id}] Session deleted.")
    return {"message": "Session deleted successfully."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
