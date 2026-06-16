"""Local FastAPI service for the MTF Mapper desktop frontend."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import mtf_mapper_py


PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_CHART = PROJECT_DIR / "samples" / "mtf_test_chart.png"

app = FastAPI(title="MTF Mapper Local API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "tauri://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_options(options: str | None) -> dict[str, Any]:
    if not options:
        return {}
    try:
        payload = json.loads(options)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Options must be a JSON object")
    return payload


async def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "input.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while chunk := await upload.read(1024 * 1024):
            tmp.write(chunk)
        return Path(tmp.name)


async def analyze_upload(upload: UploadFile, options: str | None, action: str) -> dict[str, Any]:
    input_path = await save_upload(upload)
    parsed_options = parse_options(options)
    try:
        if action == "original":
            return mtf_mapper_py.json_safe(mtf_mapper_py.web_load_original(str(input_path), parsed_options))  # type: ignore[return-value]
        if action == "preview":
            return mtf_mapper_py.json_safe(mtf_mapper_py.web_preview_detection(str(input_path), parsed_options))  # type: ignore[return-value]
        if action == "analyze":
            return mtf_mapper_py.json_safe(mtf_mapper_py.web_analyze(str(input_path), parsed_options))  # type: ignore[return-value]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        input_path.unlink(missing_ok=True)
    raise HTTPException(status_code=400, detail=f"Unsupported action {action}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sample")
def sample_image() -> FileResponse:
    return FileResponse(SAMPLE_CHART, media_type="image/png", filename=SAMPLE_CHART.name)


@app.post("/api/original")
async def original(file: UploadFile = File(...), options: str | None = Form(None)) -> dict[str, Any]:
    return await analyze_upload(file, options, "original")


@app.post("/api/preview-detection")
async def preview_detection(file: UploadFile = File(...), options: str | None = Form(None)) -> dict[str, Any]:
    return await analyze_upload(file, options, "preview")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), options: str | None = Form(None)) -> dict[str, Any]:
    return await analyze_upload(file, options, "analyze")


def main() -> None:
    import uvicorn

    uvicorn.run("mtf_mapper_api:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
