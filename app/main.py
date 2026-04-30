# app/main.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import types
import importlib.machinery

if "librosa" not in sys.modules:
    fake = types.ModuleType("librosa")
    fake.__version__ = "0.0"
    fake.__spec__ = importlib.machinery.ModuleSpec("librosa", loader=None)
    sys.modules["librosa"] = fake

if "soundfile" not in sys.modules:
    fake = types.ModuleType("soundfile")
    fake.__version__ = "0.0"
    fake.__spec__ = importlib.machinery.ModuleSpec("soundfile", loader=None)
    sys.modules["soundfile"] = fake

import re
import csv
import time
import difflib
import shutil
from io import StringIO
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import faiss
import cv2
import yt_dlp
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DOCS_PATH   = BASE_DIR / "rag" / "retrieval_docs.json"
HF_DIR      = BASE_DIR / "hf_models"
OUT_DIR     = BASE_DIR / "downloaded_videos"
UPLOAD_DIR  = BASE_DIR / "uploaded_files"
MODEL_ID    = os.getenv("VLM_MODEL", "skt/A.X-4.0-VL-Light")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

DEFAULT_PROMPT = """
이 이미지는 동영상의 한 프레임이다. 다음 두 가지 작업을 수행하라.

1. [OCR]: 화면에 보이는 모든 한국어/영어 텍스트를 추출하라.
    - 규칙: 원문 그대로 적고, 줄바꿈을 유지하며, 읽기 어려운 부분은 [불명확]으로 표시하라. 텍스트가 없으면 "텍스트 없음"이라 한다.

2. [OBJECTS]: 이미지 내의 주요 사물들 5개를 영어로 나열하라.
    - 규칙: 콤마(,)로 구분하여 최대 5개만 영어로 나열한다. 없으면 "None"이라 한다.

두 섹션을 반드시 [OCR]과 [OBJECTS] 제목으로 구분하라.
""".strip()

# ──────────────────────────────────────────────
# 전역 모델 상태
# ──────────────────────────────────────────────
_processor = None
_model     = None
_rag       = None


# ──────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────
import json as _json

class CrimeRAG:
    def __init__(self, docs_path: Path, embed_model: str = "BAAI/bge-m3"):
        with open(docs_path, "r", encoding="utf-8") as f:
            self.docs = _json.load(f)
        self.model = SentenceTransformer(embed_model)
        self.index = None

    def build_index(self):
        texts = [d["text"] for d in self.docs]
        embs  = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def search(self, query: str, top_k: int = 3):
        if not self.index:
            self.build_index()
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.docs):
                doc = self.docs[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
        return results


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def _load_model(model_id: str):
    safe_name = model_id.replace("/", "_").replace(".", "_")
    local_dir = HF_DIR / safe_name
    if not local_dir.exists():
        snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    processor = AutoProcessor.from_pretrained(str(local_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir), torch_dtype=TORCH_DTYPE, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return processor, model


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def _format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _sample_frames(video_path: Path, every_n: float, max_f: int):
    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_n)), 1)
    frames, idx = [], 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append((idx / fps, Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))))
            if len(frames) >= max_f:
                break
        idx += 1
    cap.release()
    return frames


def _analyze_frame(image: Image.Image) -> dict:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image.convert("RGB")},
        {"type": "text", "text": DEFAULT_PROMPT},
    ]}]
    text_input = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = _processor(text=[text_input], images=[image.convert("RGB")], return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output_ids = _model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=1.0, top_p=0.95)
    new_ids   = output_ids[:, inputs["input_ids"].shape[1]:]
    text_resp = _processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    ocr_text, obj_list = "텍스트 없음", []
    ocr_m = re.search(r"\[?OCR\]?[:\s\n]+(.*?)(?=\[?OBJECTS\]?|$)", text_resp, re.S | re.I)
    if ocr_m:
        ocr_text = ocr_m.group(1).strip()
    obj_m = re.search(r"\[?OBJECTS\]?[:\s\n]+(.*?)$", text_resp, re.S | re.I)
    if obj_m:
        raw = obj_m.group(1).strip()
        if raw and "없음" not in raw:
            obj_list = [o.strip() for o in re.split(r"[,|\n·\-\.]", raw) if o.strip()]
    return {"ocr_text": ocr_text, "object_list": obj_list}


def _merge_and_rag(raw_results, top_k: int):
    all_objs, merged_ocr = set(), []
    for r in raw_results:
        all_objs.update(r["objects"])
        if not r["text"] or r["text"] in ["텍스트 없음", "[불명확]"]:
            continue
        if not merged_ocr or difflib.SequenceMatcher(None, merged_ocr[-1]["text"], r["text"]).ratio() < 0.9:
            merged_ocr.append({"start": r["ts"], "end": r["ts"], "text": r["text"]})
        else:
            merged_ocr[-1]["end"] = r["ts"]

    ocr_all = " | ".join(m["text"] for m in merged_ocr)
    obj_all = ", ".join(all_objs)
    rag_results = _rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=top_k)
    return list(all_objs), merged_ocr, rag_results


def _run_frames(frames, top_k: int):
    raw = []
    for sec, img in frames:
        res = _analyze_frame(img)
        raw.append({"sec": round(sec, 3), "ts": _format_ts(sec), "text": res["ocr_text"], "objects": res["object_list"]})
    return _merge_and_rag(raw, top_k)


# ──────────────────────────────────────────────
# 앱 생명주기 (모델 1회 로드)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model, _rag
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("[startup] 모델 로딩 중...")
    _processor, _model = await run_in_threadpool(_load_model, MODEL_ID)
    print("[startup] RAG 인덱스 구성 중...")
    _rag = await run_in_threadpool(CrimeRAG, DOCS_PATH)
    await run_in_threadpool(_rag.build_index)
    print("[startup] 준비 완료.")
    yield


app = FastAPI(title="CyberCOP Video Analysis API", lifespan=lifespan)


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────

@app.post("/api/video")
async def analyze_url(
    url:        str   = Form(..., description="YouTube / TikTok 영상 URL"),
    label:      str   = Form("manual", description="레이블 (abnormal / normal / manual)"),
    sample_sec: float = Form(1.0,  description="프레임 샘플링 간격(초)"),
    max_frames: int   = Form(10,   description="최대 프레임 수"),
    top_k:      int   = Form(3,    description="RAG 검색 상위 k"),
):
    """
    YouTube / TikTok URL → 사이버범죄 분석 JSON 반환
    """
    def _process():
        outtmpl = str(OUT_DIR / "%(title).80s_%(id)s.%(ext)s")
        with yt_dlp.YoutubeDL({"outtmpl": outtmpl, "format": "best[ext=mp4]/best",
                                "noplaylist": True, "quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=True)
        v_path = Path(ydl.prepare_filename(info))
        if not v_path.exists():
            v_path = list(OUT_DIR.glob(f"*{info['id']}*"))[0]

        video_id = info.get("id", str(int(time.time())))
        frames   = _sample_frames(v_path, sample_sec, max_frames)
        objects, merged_ocr, rag_results = _run_frames(frames, top_k)
        return {
            "id":      video_id,
            "title":   info.get("title"),
            "label":   label,
            "objects": objects,
            "ocr":     merged_ocr,
            "rag":     rag_results,
        }

    try:
        payload = await run_in_threadpool(_process)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"message": "success", "result": payload})


@app.post("/api/video/upload")
async def analyze_upload(
    file:       UploadFile = File(..., description="영상 또는 이미지 파일"),
    label:      str        = Form("manual"),
    sample_sec: float      = Form(1.0),
    max_frames: int        = Form(10),
    top_k:      int        = Form(3),
):
    """
    영상 / 이미지 파일 업로드 → 사이버범죄 분석 JSON 반환
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")
    ext = Path(file.filename).suffix.lower()
    if ext not in VIDEO_EXTS | IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다. ({ext})")

    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    def _process():
        if ext in IMAGE_EXTS:
            frames = [(0.0, Image.open(save_path).convert("RGB"))]
        else:
            frames = _sample_frames(save_path, sample_sec, max_frames)
        objects, merged_ocr, rag_results = _run_frames(frames, top_k)
        return {
            "id":      save_path.stem,
            "title":   file.filename,
            "label":   label,
            "objects": objects,
            "ocr":     merged_ocr,
            "rag":     rag_results,
        }

    try:
        payload = await run_in_threadpool(_process)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"message": "success", "result": payload})


@app.post("/api/video/csv")
async def analyze_csv(
    file:       UploadFile = File(..., description="link, label 컬럼을 가진 CSV 파일"),
    sample_sec: float      = Form(1.0),
    max_frames: int        = Form(10),
    top_k:      int        = Form(3),
):
    """
    CSV 파일 배치 업로드 (link, label 컬럼) → 전체 분석 결과 JSON 반환
    """
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="CSV 파일(.csv)만 지원합니다.")

    content = (await file.read()).decode("utf-8")
    reader  = csv.DictReader(StringIO(content))
    rows    = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")
    if "link" not in rows[0]:
        raise HTTPException(status_code=400, detail="CSV에 'link' 컬럼이 필요합니다.")

    def _process_all():
        results = []
        for row in rows:
            url   = row["link"]
            label = row.get("label", "unknown")
            try:
                outtmpl = str(OUT_DIR / "%(title).80s_%(id)s.%(ext)s")
                with yt_dlp.YoutubeDL({"outtmpl": outtmpl, "format": "best[ext=mp4]/best",
                                       "noplaylist": True, "quiet": True, "no_warnings": True}) as ydl:
                    info = ydl.extract_info(url, download=True)
                v_path = Path(ydl.prepare_filename(info))
                if not v_path.exists():
                    v_path = list(OUT_DIR.glob(f"*{info['id']}*"))[0]
                video_id = info.get("id", str(int(time.time())))
                frames   = _sample_frames(v_path, sample_sec, max_frames)
                objects, merged_ocr, rag_results = _run_frames(frames, top_k)
                results.append({
                    "id":      video_id,
                    "title":   info.get("title"),
                    "label":   label,
                    "objects": objects,
                    "ocr":     merged_ocr,
                    "rag":     rag_results,
                    "error":   None,
                })
            except Exception as e:
                results.append({"url": url, "label": label, "error": str(e)})
        return results

    try:
        results = await run_in_threadpool(_process_all)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"message": "success", "count": len(results), "results": results})
