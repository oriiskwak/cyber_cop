# -*- coding: utf-8 -*-
# video_ocr_minicpmo45_awq_rag.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import types
import importlib.machinery
import re
import json
import time
import shutil
import argparse
import difflib
import csv
from pathlib import Path

# Windows 콘솔 인코딩 문제 해결 (유니코드 출력 지원)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", line_buffering=True)

# OCR stub
if "librosa" not in sys.modules:
    fake_librosa = types.ModuleType("librosa")
    fake_librosa.__version__ = "0.0"
    fake_librosa.__spec__ = importlib.machinery.ModuleSpec("librosa", loader=None)
    sys.modules["librosa"] = fake_librosa

if "soundfile" not in sys.modules:
    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.__version__ = "0.0"
    fake_soundfile.__spec__ = importlib.machinery.ModuleSpec("soundfile", loader=None)
    sys.modules["soundfile"] = fake_soundfile

import cv2
import yt_dlp
import torch
import numpy as np
import faiss
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

MODEL_ID = "openbmb/MiniCPM-o-4_5-awq"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_DOCS_PATH = "./rag/retrieval_docs.json"

# 모델 필터링 패턴
ALLOW_PATTERNS = ["*.json", "*.txt", "*.py", "*.safetensors", "*.bin", "*.model", "*.tiktoken"]
IGNORE_PATTERNS = ["assets/*", "**/assets/*", "README*", "readme*", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.wav", "*.mp3", "*.mp4", "*.webm", "*.mov", "*.avi", "*.mkv", "*.yaml", "*.yml"]

def prepare_local_model(model_id: str, local_root: Path) -> Path:
    local_dir = local_root / model_id.split("/")[-1]
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    return local_dir.resolve()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# [보완] 범죄유형별 위험도 수치 맵 (이미지 데이터 기반)
CRIME_RISK_MAP = {
    "직거래 사기": {"months": 12.0, "risk": 0.22},
    "쇼핑몰 사기": {"months": 12.0, "risk": 0.22},
    "게임 사기": {"months": 12.0, "risk": 0.22},
    "이메일 무역사기": {"months": 30.0, "risk": 0.55},
    "기타 사이버 사기": {"months": 12.0, "risk": 0.22},
    "피싱": {"months": 12.0, "risk": 0.22},
    "파밍": {"months": 12.0, "risk": 0.22},
    "스미싱": {"months": 12.0, "risk": 0.22},
    "메모리 해킹": {"months": 18.0, "risk": 0.33},
    "몸캠피싱": {"months": 39.0, "risk": 0.71},
    "메신저 이용사기": {"months": 12.0, "risk": 0.22},
    "기타 사이버 금융범죄": {"months": 10.0, "risk": 0.18},
    "개인·위치정보 침해": {"months": 12.3, "risk": 0.22},
    "사이버저작권 침해": {"months": 13.0, "risk": 0.24},
    "기타 정보통신망 이용범죄": {"months": 7.0, "risk": 0.13},
    "아동성 착취물": {"months": 55.0, "risk": 1.00},
    "불법 촬영물": {"months": 24.3, "risk": 0.44},
    "허위 영상물": {"months": 15.0, "risk": 0.27},
    "불법성 영상물": {"months": 7.0, "risk": 0.13},
    "스포츠 토토": {"months": 16.0, "risk": 0.29},
    "경마·경륜·경정": {"months": 13.0, "risk": 0.24},
    "카지노": {"months": 13.0, "risk": 0.24},
    "기타 사이버 도박": {"months": 13.0, "risk": 0.24},
    "명예훼손": {"months": 11.0, "risk": 0.20},
    "모욕": {"months": 5.0, "risk": 0.09},
    "기타 불법콘텐츠 범죄": {"months": 8.0, "risk": 0.15}
}

# 속도 최적화된 프롬프트: 365개 리스트 제거
DEFAULT_PROMPT = """
이 이미지는 동영상의 한 프레임이다. 다음 두 가지 작업을 수행하라.

1. [OCR]: 화면에 보이는 모든 한국어/영어 텍스트를 추출하라.
    - 규칙: 원문 그대로 적고, 줄바꿈을 유지하며, 읽기 어려운 부분은 [불명확]으로 표시하라. 텍스트가 없으면 "텍스트 없음"이라 한다.

2. [OBJECTS]: 이미지 내의 주요 사물들 5개를 영어로 나열하라.
    - 규칙: 콤마(,)로 구분하여 최대 5개만 영어로 나열한다. 없으면 "None"이라 한다.

두 섹션을 반드시 [OCR]과 [OBJECTS] 제목으로 구분하라.
""".strip()

class CrimeRAG:
    def __init__(self, docs_path: str, model_name: str = DEFAULT_EMBED_MODEL):
        self.docs_path = Path(docs_path)
        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.model = SentenceTransformer(model_name)
        self.index = None

    def build_index(self):
        texts = [doc["text"] for doc in self.docs]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        if not self.index:
            self.build_index()
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.docs): continue
            doc = self.docs[idx].copy()
            doc["similarity"] = float(score)
            
            # 위험도 매핑 로직 추가
            crime_type = doc.get("crime_type", "기타")
            risk_data = CRIME_RISK_MAP.get(crime_type, {"months": 0.0, "risk": 0.0})
            doc["risk_level"] = risk_data["risk"]
            
            results.append(doc)
        return results

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def download_video(url: str, out_dir: Path) -> tuple[Path, dict]:
    outtmpl = str(out_dir / "%(title).80s_%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    video_path = Path(ydl.prepare_filename(info))
    if not video_path.exists():
        video_path = list(out_dir.glob(f"*{info['id']}*"))[0]
    return video_path, info

def sample_frames(video_path: Path, every_n: float, max_f: int) -> list[tuple[float, Image.Image]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_n)), 1)
    frames, idx = [], 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if idx % step == 0:
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frames.append((idx / fps, image))
            if len(frames) >= max_f: break
        idx += 1
    cap.release()
    return frames

def analyze_one_frame(model, tokenizer, image: Image.Image, prompt: str) -> dict:
    msgs = [{"role": "user", "content": prompt}]
    with torch.inference_mode():
        response = model.chat(image=image.convert("RGB"), msgs=msgs, tokenizer=tokenizer, max_new_tokens=1024, sampling=True, temperature=0.7)
    text_resp = str(response[0] if isinstance(response, (list, tuple)) else response).strip()
    
    # 디버깅을 위한 원문 출력
    print(f"\n--- RAW RESPONSE ---\n{text_resp}\n---------------------")
    
    ocr_text, obj_list = "텍스트 없음", []
    
    # 1. OCR 추출 ([OCR] 태그 기반)
    ocr_match = re.search(r"\[?OCR\]?[:\s\n]+(.*?)(?=\[?OBJECTS\]?|$)", text_resp, re.S | re.I)
    if ocr_match: 
        ocr_text = ocr_match.group(1).strip()

    # 2. 객체 추출 ([OBJECTS] 태그 기반)
    obj_match = re.search(r"\[?OBJECTS\]?[:\s\n]+(.*?)$", text_resp, re.S | re.I)
    if obj_match:
        raw = obj_match.group(1).strip()
        if raw and "없음" not in raw:
            # 쉼표, 줄바꿈, 점 등으로 분리
            obj_list = [o.strip() for o in re.split(r"[,|\n·\-\.]", raw) if o.strip()]
            
    return {"ocr_text": ocr_text, "object_list": obj_list}

def process_single_video(url: str, label: str, model_pack, rag, args):
    video_id = re.search(r"(?:v=|video/|shorts/)([a-zA-Z0-9_-]+)", url)
    video_id = video_id.group(1) if video_id else str(int(time.time()))
    
    out_dir = Path(args.out_dir)
    res_dir = ensure_dir(out_dir / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"
    txt_path = res_dir / f"ocr_merged_{video_id}.txt"

    if json_path.exists():
        print(f"  [Skip] Already completed: {video_id}")
        return

    print(f"  [Processing] {url} (Label: {label})")
    try:
        v_path, info = download_video(url, out_dir)
        frames = sample_frames(v_path, args.sample_sec, args.max_frames)
        
        raw_results, all_objs = [], set()
        for i, (sec, img) in enumerate(frames, 1):
            print(f"    - Frame {i}/{len(frames)} ({format_ts(sec)})")
            res = analyze_one_frame(model_pack[0], model_pack[1], img, DEFAULT_PROMPT)
            raw_results.append({"sec": round(sec, 3), "ts": format_ts(sec), "text": res["ocr_text"], "objects": res["object_list"]})
            all_objs.update(res["object_list"])

        # Merge OCR
        merged_ocr = []
        for r in raw_results:
            if not r["text"] or r["text"] in ["텍스트 없음", "[불명확]"]: continue
            if not merged_ocr or difflib.SequenceMatcher(None, merged_ocr[-1]["text"], r["text"]).ratio() < 0.9:
                merged_ocr.append({"text": r["text"]})

        # RAG Search
        ocr_all = " | ".join([m["text"] for m in merged_ocr])
        obj_all = ", ".join(list(all_objs))
        mapped = rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=args.top_k)

        # Output payload
        payload = {"id": video_id, "title": info.get("title"), "label": label, "objects": list(all_objs), "ocr": merged_ocr, "rag": mapped}
        with open(json_path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"VIDEO: {info.get('title')}\nURL: {url}\nLABEL: {label}\n\n[Mapped Crimes]\n")
            for c in mapped: f.write(f"- {c['crime_type']} ({c['similarity']:.4f})\n")
            f.write(f"\n[Detected Objects]\n{obj_all}\n\n[OCR]\n")
            for m in merged_ocr: f.write(f"- {m['text']}\n")
        print(f"  [Done] Result saved: {json_path}")
    except Exception as e:
        print(f"  [Error] {url}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--csv")
    parser.add_argument("--out_dir", default="./downloaded_videos")
    parser.add_argument("--hf_dir", default="./hf_models")
    parser.add_argument("--sample_sec", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    print("[1/3] Loading Model (Flash Mode)...")
    # 모델 유무 확인 및 자동 확보
    local_root = ensure_dir(args.hf_dir)
    local_dir = prepare_local_model(MODEL_ID, local_root)
    
    print(f"  - Using local path: {local_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(local_dir), trust_remote_code=True, torch_dtype=torch.float16).to("cuda:0")
    model.eval()
    
    print("[2/3] Preparing RAG Index...")
    rag = CrimeRAG(DEFAULT_DOCS_PATH)
    
    print("[3/3] Starting Batch Analysis...")
    if args.url:
        process_single_video(args.url, "manual", (model, tokenizer), rag, args)
    elif args.csv:
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                process_single_video(row["link"], row.get("label", "unknown"), (model, tokenizer), rag, args)
    else:
        print("URL or CSV path required.")

if __name__ == "__main__":
    main()
