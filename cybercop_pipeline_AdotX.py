import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import types
import importlib.machinery
import re
import json
import time
import base64
import argparse
import difflib
import csv
from io import BytesIO
from pathlib import Path

# Windows 콘솔 인코딩 문제 해결 (유니코드 출력 지원)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", line_buffering=True)

# OCR stub (librosa / soundfile 미사용이지만 하위 호환 유지)
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
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

MODEL_ID            = os.getenv("VLM_MODEL", "skt/A.X-4.0-VL-Light")
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE         = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_DOCS_PATH   = "./rag/retrieval_docs.json"

# Objects365 한국어 허용 클래스명 리스트 (검증용)
ALLOWED_OBJECTS = [
    "사람", "운동화", "의자", "기타 신발", "모자", "자동차", "램프", "안경", "병", "책상",
    "컵", "가로등", "캐비닛/선반", "핸드백/가방", "팔찌", "접시", "그림/액자", "헬멧", "책", "장갑",
    "수납상자", "보트", "가죽구두", "꽃", "벤치", "화분식물", "그릇/대야", "깃발", "베개", "부츠",
    "꽃병", "마이크", "목걸이", "반지", "SUV", "와인잔", "벨트", "모니터/TV", "백팩", "우산",
    "신호등", "스피커", "손목시계", "넥타이", "쓰레기통", "슬리퍼", "자전거", "스툴", "통/양동이", "밴",
    "소파", "샌들", "바구니", "드럼", "펜/연필", "버스", "야생 조류", "하이힐", "오토바이", "기타",
    "카펫", "휴대전화", "빵", "카메라", "캔류", "트럭", "라바콘", "심벌즈", "구명부표", "수건",
    "봉제인형", "양초", "범선", "노트북", "차양막", "침대", "수도꼭지", "텐트", "말", "거울",
    "전원 콘센트", "싱크대", "사과", "에어컨", "칼", "하키 스틱", "패들", "픽업트럭", "포크", "교통표지판",
    "풍선", "삼각대", "개", "숟가락", "시계", "냄비", "소", "케이크", "식탁", "양",
    "옷걸이", "칠판/화이트보드", "냅킨", "기타 물고기", "오렌지/귤", "세면도구", "키보드", "토마토", "랜턴", "작업차량",
    "선풍기", "녹색 채소", "바나나", "야구 글러브", "비행기", "마우스", "기차", "호박", "축구공", "스키보드",
    "여행가방", "협탁", "찻주전자", "전화기", "카트", "헤드폰", "스포츠카", "정지표지판", "디저트", "스쿠터",
    "유모차", "크레인", "리모컨", "냉장고", "오븐", "레몬", "오리", "야구배트", "감시카메라", "고양이",
    "저그", "브로콜리", "피아노", "피자", "코끼리", "스케이트보드", "서핑보드", "총", "스케이트/스키 신발", "가스레인지",
    "도넛", "보타이", "당근", "변기", "연", "딸기", "기타 공", "삽", "고추", "컴퓨터 본체",
    "화장지", "청소용품", "젓가락", "전자레인지", "비둘기", "야구공", "도마", "커피테이블", "사이드테이블", "가위",
    "마커", "파이", "사다리", "스노보드", "쿠키", "라디에이터", "소화전", "농구공", "얼룩말", "포도",
    "기린", "감자", "소시지", "세발자전거", "바이올린", "달걀", "소화기", "사탕", "소방차", "당구",
    "컨버터", "욕조", "휠체어", "골프채", "서류가방", "오이", "시가/담배", "붓", "배", "대형트럭",
    "햄버거", "환풍기", "연장선", "집게", "테니스 라켓", "폴더", "미식축구공", "이어폰", "마스크", "주전자",
    "테니스공", "선박", "그네", "커피머신", "미끄럼틀", "마차", "양파", "그린빈", "프로젝터", "프리스비",
    "세탁기/건조기", "닭", "프린터", "수박", "색소폰", "티슈", "칫솔", "아이스크림", "열기구", "첼로",
    "감자튀김", "저울", "트로피", "데이터복구", "양배추", "핫도그", "블렌더", "복숭아", "밥/쌀", "지갑", "배구공",
    "사슴", "거위", "테이프", "태블릿", "화장품", "트럼펫", "파인애플", "골프공", "구급차", "주차미터기",
    "망고", "열쇠", "허들", "낚싯대", "메달", "플루트", "브러시", "펭귄", "메가폰", "옥수수",
    "상추", "마늘", "백조", "헬리콥터", "대파", "샌드위치", "견과류", "속도제한표지판", "인덕션", "빗자루",
    "트롬본", "자두", "인력거", "금붕어", "키위", "라우터/모뎀", "포커카드", "토스터", "새우", "초밥",
    "치즈", "메모지", "체리", "펜치", "CD", "파스타", "망치", "큐대", "아보카도", "하미멜론",
    "플라스크", "버섯", "드라이버", "비누", "리코더", "곰", "가지", "칠판지우개", "코코넛", "줄자/자",
    "돼지", "샤워기", "지구본", "칩", "스테이크", "횡단보도표지판", "스테이플러", "낙타", "포뮬러원 경주차", "석류",
    "식기세척기", "게", "호버보드", "미트볼", "밥솥", "튜바", "계산기", "파파야", "영양", "앵무새",
    "물개", "나비", "덤벨", "당나귀", "사자", "소변기", "돌고래", "전동드릴", "헤어드라이어", "에그타르트",
    "해파리", "러닝머신", "라이터", "자몽", "게임판", "대걸레", "무", "바오쯔", "표적", "프렌치",
    "춘권", "원숭이", "토끼", "필통", "야크", "적양배추", "쌍안경", "아스파라거스", "바벨", "가리비",
    "면류", "빗", "만두", "굴", "탁구채", "화장붓/아이라이너 펜슬", "전기톱", "지우개", "바닷가재", "두리안",
    "오크라", "립스틱", "손거울", "컬링", "탁구"
]

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

# ──────────────────────────────────────────────

# 프롬프트
# ──────────────────────────────────────────────
DEFAULT_PROMPT = """
이 이미지는 동영상의 한 프레임이다. 다음 두 가지 작업을 수행하라.

1. [OCR]: 화면에 보이는 모든 한국어/영어 텍스트를 추출하라.
    - 규칙: 원문 그대로 적고, 줄바꿈을 유지하며, 읽기 어려운 부분은 [불명확]으로 표시하라. 텍스트가 없으면 "텍스트 없음"이라 한다.

2. [OBJECTS]: 이미지 내에 보이는 사물들을 한국어로 자유롭게 나열하라.
    - 규칙:
      - 콤마(,)로 구분하여 최대 10개만 나열한다.
      - 설명문 없이 사물명만 적는다.
      - 같은 사물을 중복해서 쓰지 않는다.
      - 확신이 낮은 객체는 제외한다.
      - 적절한 객체가 없으면 "없음"이라 한다.
      - 예시: 사람, 의자, 자동차
      - 잘못된 예시: Person, chair, tv, table_tennis

두 섹션을 반드시 [OCR]과 [OBJECTS] 제목으로 구분하라.
""".strip()



# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_model(model_id: str):
    print(f"  [INFO] 모델 로드 중: {model_id}  (device={DEVICE}, dtype={TORCH_DTYPE})")
    # 모델 이름에 점(.)이 포함되면 transformers 모듈 경로가 꼬이므로 로컬에 먼저 다운로드
    safe_name = model_id.replace("/", "_").replace(".", "_")
    local_dir = Path("./hf_models") / safe_name
    if not local_dir.exists():
        print(f"  [INFO] 로컬 캐시 없음 — HuggingFace에서 다운로드 중...")
        snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    print(f"  [INFO] 로컬 경로에서 로드: {local_dir}")
    processor = AutoProcessor.from_pretrained(
        str(local_dir),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(local_dir),
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("  [OK] 모델 로드 완료.")
    return processor, model


# ──────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────
class CrimeRAG:
    def __init__(self, docs_path: str, model_name: str = DEFAULT_EMBED_MODEL):
        self.docs_path = Path(docs_path)
        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.model = SentenceTransformer(model_name)
        self.index = None

    def build_index(self):
        texts = [doc["text"] for doc in self.docs]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        if not self.index:
            self.build_index()
        q_emb = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx].copy()
            doc["similarity"] = float(score)
            
            # 위험도 매핑 로직 추가
            crime_type = doc.get("crime_type", "기타")
            risk_data = CRIME_RISK_MAP.get(crime_type, {"months": 0.0, "risk": 0.0})
            doc["risk_level"] = risk_data["risk"]
            
            results.append(doc)
        return results


# ──────────────────────────────────────────────
# 객체 매퍼 (임베딩 기반 ALLOWED_OBJECTS 매핑)
# ──────────────────────────────────────────────
class ObjectMapper:
    def __init__(self, embed_model: SentenceTransformer, threshold: float = 0.6):
        self.model = embed_model
        self.threshold = threshold
        self.allowed = ALLOWED_OBJECTS
        self.embeddings = self.model.encode(
            self.allowed, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

    def map(self, candidates: list) -> list:
        if not candidates:
            return []
        cand_embs = self.model.encode(
            candidates, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores = cand_embs @ self.embeddings.T
        result, seen = [], set()
        for cand_scores in scores:
            best_idx = int(cand_scores.argmax())
            best_score = float(cand_scores[best_idx])
            if best_score >= self.threshold:
                mapped = self.allowed[best_idx]
                if mapped not in seen:
                    seen.add(mapped)
                    result.append(mapped)
        return result


# ──────────────────────────────────────────────
# 영상 처리
# ──────────────────────────────────────────────
def download_video(url: str, out_dir: Path) -> tuple[Path, dict]:
    outtmpl = str(out_dir / "%(title).80s_%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
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
        if not ok:
            break
        if idx % step == 0:
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frames.append((idx / fps, image))
            if len(frames) >= max_f:
                break
        idx += 1
    cap.release()
    return frames


# ──────────────────────────────────────────────
# 프레임 분석 (transformers)
# ──────────────────────────────────────────────
def analyze_one_frame(
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    image: Image.Image,
    prompt: str,
    obj_mapper: "ObjectMapper",
    max_new_tokens: int = 1024,
) -> dict:
    # PIL 이미지를 messages 안에 직접 포함해야 apply_chat_template이 이미지 토큰을 삽입함
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text_input],
        images=[image.convert("RGB")],
        return_tensors="pt",
    ).to(DEVICE)

    # 생성
    start_time = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
        )

    # 입력 토큰 제거 후 디코딩
    input_len   = inputs["input_ids"].shape[1]
    new_ids     = output_ids[:, input_len:]
    text_resp   = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    elapsed     = time.time() - start_time
    # 디버깅용 원문 출력
    print(f"\n--- RAW RESPONSE ({elapsed:.2f}s) ---\n{text_resp}\n---------------------")

    ocr_text, obj_list = "텍스트 없음", []

    # OCR 추출
    ocr_match = re.search(
        r"\[?OCR\]?[:\s\n]+(.*?)(?=\[?OBJECTS\]?|$)", text_resp, re.S | re.I
    )
    if ocr_match:
        ocr_text = ocr_match.group(1).strip()

    # 객체 추출 및 임베딩 기반 매핑
    obj_match = re.search(r"\[?OBJECTS\]?[:\s\n]+(.*?)$", text_resp, re.S | re.I)
    if obj_match:
        raw = obj_match.group(1).strip()
        if raw and "없음" not in raw:
            candidates = [o.strip() for o in re.split(r"[,|\n·\-\.]", raw) if o.strip()]
            obj_list = obj_mapper.map(candidates)
            print(f"    [Post-Process] {candidates} → {obj_list}")

    return {"ocr_text": ocr_text, "object_list": obj_list, "inference_time": elapsed}


# ──────────────────────────────────────────────
# 단일 영상 처리 파이프라인
# ──────────────────────────────────────────────
def process_single_video(
    url: str,
    label: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    rag: CrimeRAG,
    obj_mapper: ObjectMapper,
    args,
):
    video_id_match = re.search(r"(?:v=|video/|shorts/)([a-zA-Z0-9_-]+)", url)
    video_id = video_id_match.group(1) if video_id_match else str(int(time.time()))

    out_dir  = Path(args.out_dir)
    res_dir  = ensure_dir(out_dir / "results")
    json_path = res_dir / f"ocr_results_{video_id}.json"
    txt_path  = res_dir / f"ocr_merged_{video_id}.txt"

    if json_path.exists():
        print(f"  [Skip] Already completed: {video_id}")
        return

    print(f"  [Processing] {url} (Label: {label})")
    video_start = time.time()
    try:
        v_path, info = download_video(url, out_dir)
        frames = sample_frames(v_path, args.sample_sec, args.max_frames)

        raw_results, all_objs = [], set()
        for i, (sec, img) in enumerate(frames, 1):
            res = analyze_one_frame(processor, model, img, DEFAULT_PROMPT, obj_mapper)
            print(f"    - Frame {i}/{len(frames)} ({format_ts(sec)}) [{res['inference_time']:.2f}s]")
            raw_results.append({
                "sec": round(sec, 3),
                "ts": format_ts(sec),
                "text": res["ocr_text"],
                "objects": res["object_list"],
            })
            all_objs.update(res["object_list"])

        # OCR 병합 (중복 제거)
        merged_ocr = []
        for r in raw_results:
            if not r["text"] or r["text"] in ["텍스트 없음", "[불명확]"]:
                continue
            if not merged_ocr or difflib.SequenceMatcher(
                None, merged_ocr[-1]["text"], r["text"]
            ).ratio() < 0.9:
                merged_ocr.append({"start": r["ts"], "end": r["ts"], "text": r["text"]})
            else:
                merged_ocr[-1]["end"] = r["ts"]

        # RAG 검색
        ocr_all = " | ".join([m["text"] for m in merged_ocr])
        obj_all = ", ".join(list(all_objs))
        mapped  = rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=args.top_k)

        # 결과 저장
        video_elapsed = time.time() - video_start
        payload = {
            "id": video_id,
            "url": url,
            "title": info.get("title"),
            "label": label,
            "objects": list(all_objs),
            "ocr": merged_ocr,
            "rag": mapped,
            "total_inference_time": round(video_elapsed, 2)
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"VIDEO: {info.get('title')}\nURL: {url}\nLABEL: {label}\n\n[Mapped Crimes]\n")
            for c in mapped:
                f.write(f"- {c['crime_type']} ({c['similarity']:.4f})\n")
            f.write(f"\n[Detected Objects]\n{obj_all}\n\n[OCR Timeline]\n")
            for m in merged_ocr:
                f.write(f"[{m['start']}~{m['end']}] {m['text']}\n")
        print(f"  [RAG] 매칭된 범죄 유형:")
        for c in mapped:
            print(f"    - {c['crime_type']} (score: {c['similarity']:.4f})")
        video_elapsed = time.time() - video_start
        print(f"  [Done] Result saved: {json_path} (Total: {video_elapsed:.2f}s)")

    except Exception as e:
        print(f"  [Error] {url}: {e}")


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


# ──────────────────────────────────────────────
# 로컬 파일 처리 (영상 or 이미지)
# ──────────────────────────────────────────────
def process_local_file(
    file_path: Path,
    label: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    rag: CrimeRAG,
    obj_mapper: ObjectMapper,
    args,
):
    file_id = file_path.stem
    ext = file_path.suffix.lower()
    res_dir = ensure_dir(Path(args.out_dir) / "results")
    json_path = res_dir / f"ocr_results_{file_id}.json"
    txt_path  = res_dir / f"ocr_merged_{file_id}.txt"

    if json_path.exists():
        print(f"  [Skip] Already completed: {file_id}")
        return

    print(f"  [Processing] {file_path.name} (Label: {label})")
    file_start = time.time()
    try:
        if ext in IMAGE_EXTS:
            image = Image.open(file_path).convert("RGB")
            frames = [(0.0, image)]
        else:
            frames = sample_frames(file_path, args.sample_sec, args.max_frames)

        raw_results, all_objs = [], set()
        for i, (sec, img) in enumerate(frames, 1):
            res = analyze_one_frame(processor, model, img, DEFAULT_PROMPT, obj_mapper)
            print(f"    - Frame {i}/{len(frames)} ({format_ts(sec)}) [{res['inference_time']:.2f}s]")
            raw_results.append({
                "sec": round(sec, 3),
                "ts": format_ts(sec),
                "text": res["ocr_text"],
                "objects": res["object_list"],
            })
            all_objs.update(res["object_list"])

        merged_ocr = []
        for r in raw_results:
            if not r["text"] or r["text"] in ["텍스트 없음", "[불명확]"]:
                continue
            if not merged_ocr or difflib.SequenceMatcher(
                None, merged_ocr[-1]["text"], r["text"]
            ).ratio() < 0.9:
                merged_ocr.append({"start": r["ts"], "end": r["ts"], "text": r["text"]})
            else:
                merged_ocr[-1]["end"] = r["ts"]

        ocr_all = " | ".join([m["text"] for m in merged_ocr])
        obj_all = ", ".join(list(all_objs))
        mapped  = rag.search(f"[OCR]: {ocr_all} | [Objects]: {obj_all}", top_k=args.top_k)

        payload = {
            "id": file_id,
            "url": str(file_path),
            "title": file_path.name,
            "label": label,
            "objects": list(all_objs),
            "ocr": merged_ocr,
            "rag": mapped,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"FILE: {file_path.name}\nLABEL: {label}\n\n[Mapped Crimes]\n")
            for c in mapped:
                f.write(f"- {c['crime_type']} ({c['similarity']:.4f})\n")
            f.write(f"\n[Detected Objects]\n{obj_all}\n\n[OCR Timeline]\n")
            for m in merged_ocr:
                f.write(f"[{m['start']}~{m['end']}] {m['text']}\n")
        print(f"  [RAG] 매칭된 범죄 유형:")
        for c in mapped:
            print(f"    - {c['crime_type']} (score: {c['similarity']:.4f})")
        file_elapsed = time.time() - file_start
        print(f"  [Done] Result saved: {json_path} (Total: {file_elapsed:.2f}s)")

    except Exception as e:
        print(f"  [Error] {file_path.name}: {e}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Video OCR + RAG pipeline (HuggingFace Transformers backend)"
    )
    parser.add_argument("--url",        help="단일 YouTube URL")
    parser.add_argument("--file",       help="단일 로컬 영상/이미지 파일 경로")
    parser.add_argument("--csv",        default="./data/labels.csv", help="link,label 컬럼을 가진 CSV 파일 경로")
    parser.add_argument("--dir",        help="영상/이미지 파일이 담긴 디렉토리 경로")
    parser.add_argument("--out_dir",    default="./downloaded_videos")
    parser.add_argument("--sample_sec", type=float, default=2.0,  help="프레임 샘플링 간격(초)")
    parser.add_argument("--max_frames", type=int,   default=10,   help="최대 프레임 수")
    parser.add_argument("--top_k",      type=int,   default=3,    help="RAG top-k")
    parser.add_argument("--model",      default=MODEL_ID,         help="HuggingFace 모델 ID")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 모델 로드
    print("[1/3] Loading VLM model via HuggingFace Transformers...")
    processor, model = load_model(args.model)

    # RAG 인덱스 + ObjectMapper
    print("[2/3] Preparing RAG Index & Object Mapper...")
    rag = CrimeRAG(DEFAULT_DOCS_PATH)
    obj_mapper = ObjectMapper(rag.model)

    # 배치 처리
    print("[3/3] Starting Batch Analysis...")
    if args.url:
        process_single_video(args.url, "manual", processor, model, rag, obj_mapper, args)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"\n[Error] 파일을 찾을 수 없습니다: {args.file}")
            sys.exit(1)
        process_local_file(file_path, "manual", processor, model, rag, obj_mapper, args)
    elif args.csv:
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                process_single_video(
                    row["link"], row.get("label", "unknown"),
                    processor, model, rag, obj_mapper, args
                )
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"\n[Error] 디렉토리를 찾을 수 없습니다: {args.dir}")
            sys.exit(1)
        files = sorted(f for f in dir_path.iterdir()
                       if f.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS)
        if not files:
            print(f"\n[Error] 지원 파일 없음 (지원 확장자: {VIDEO_EXTS | IMAGE_EXTS})")
            sys.exit(1)
        print(f"  총 {len(files)}개 파일 발견")
        for f in files:
            process_local_file(f, "manual", processor, model, rag, obj_mapper, args)
    else:
        print(f"\n[Error] --url / --csv / --dir 중 하나가 필요합니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()