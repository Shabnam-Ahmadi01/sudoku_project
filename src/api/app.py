import os
import sys
import tempfile
import warnings
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import keras
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import MODEL_PATH, CONFIDENCE_THRESHOLD
# Import custom layer classes so @register_keras_serializable fires before load_model
from src.recognition.model import (  # noqa: F401
    RandomGaussianBlur, RandomMotionBlur, RandomPerspectiveTransform,
    RandomErodeDilate, RandomJPEGCompression,
)
from src.preprocessing import process_image
from src.preprocessing.cell_extractor import split_into_cells, is_cell_empty
from src.recognition import predict_from_cells
from src.solver import solve_sudoku

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
    yield


app = FastAPI(title="Sudoku Solver API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    try:
        # Phase 1 — grid detection (once; rotation happens on the warped image)
        process_result = process_image(image_path)
        warped_gray = process_result.warped_gray

        first_failure = None

        # Try original orientation then 90°, 180°, 270° CCW rotations
        for k in range(4):
            rotated = np.rot90(warped_gray, k=k)

            cells = split_into_cells(rotated)
            empty_mask = [is_cell_empty(c) for c in cells]

            # Phase 2 — OCR on rotated cells
            recognized, confidences = predict_from_cells(
                model, cells, empty_mask, CONFIDENCE_THRESHOLD
            )

            # Phase 3 — solve
            solved, conflicts = solve_sudoku(recognized, raise_on_invalid=False)

            if not conflicts and solved is not None:
                # Unrotate both matrices back to the original image orientation
                unrot = (4 - k) % 4
                rec_out = np.rot90(recognized, k=unrot).tolist()
                sol_out = np.rot90(np.array(solved), k=unrot).tolist()
                return {
                    "status": "success",
                    "recognized_matrix": rec_out,
                    "solution": sol_out,
                    "mean_confidence": float(confidences.mean()) if len(confidences) else 0.0,
                    "confidences": confidences.tolist(),
                    "rotation_attempts": k,
                }

            if first_failure is None:
                first_failure = {
                    "recognized": recognized,
                    "confidences": confidences,
                    "conflicts": conflicts,
                }

        # All four orientations failed — report the result from the first attempt
        rec = first_failure["recognized"]
        conf = first_failure["confidences"]
        conflicts = first_failure["conflicts"]
        mean_conf = float(conf.mean()) if len(conf) else 0.0

        if conflicts:
            return JSONResponse(status_code=422, content={
                "status": "recognition_invalid",
                "recognized_matrix": rec.tolist(),
                "conflicts": conflicts,
                "mean_confidence": mean_conf,
            })

        return JSONResponse(status_code=422, content={
            "status": "unsolvable",
            "recognized_matrix": rec.tolist(),
            "mean_confidence": mean_conf,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)