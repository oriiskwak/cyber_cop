# Video OCR & RAG 기반 사이버 범죄 탐지 파이프라인

YouTube/TikTok 영상 또는 로컬 영상/이미지에서 프레임을 추출하고, VLM(SKT A.X-4.0-VL-Light)으로 OCR 및 객체 인식을 수행합니다.
추출된 정보를 바탕으로 RAG(FAISS + BGE-m3)로 사이버 범죄 유형을 매칭하고 위험도를 산출합니다.

## 시스템 요구사항

- **Python**: 3.10
- **CUDA**: 12.4 (GPU 필수)
- **VRAM**: 10GB 이상
- **테스트 환경**: torch 2.6.0 + transformers 4.51.3

## 디렉토리 구조

```
├── cybercop_pipeline_AdotX.py     # 메인 실행 파일 (SKT A.X VLM)
├── cybercop_pipeline_gpt.py       # GPT 기반 파이프라인
├── requirements.txt               # 패키지 목록
├── install.bat                    # Windows 설치 스크립트
├── data/
│   ├── labels.csv                 # 테스트용 영상 URL 목록
│   └── 7.png                      # 테스트용 이미지 샘플
├── rag/
│   ├── allowed_objects.json       # 허용 객체 클래스 목록 (446종, JSON DB)
│   └── retrieval_docs.json        # 범죄 유형 문서 (26종 + 객체 키워드 자동 추가분)
├── downloaded_videos/
│   └── results/                   # 결과 JSON/TXT 저장 (자동 생성)
└── hf_models/                     # 모델 캐시 (자동 생성)
```

## 설치

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
```

> Linux GPU 서버: `requirements.txt`의 `faiss-cpu` → `faiss-gpu` 변경 후 설치

## 실행 방법

### 단일 YouTube/TikTok 영상
```bash
python cybercop_pipeline_AdotX.py --url "https://www.youtube.com/shorts/영상ID"
```

### 단일 로컬 파일 (영상 또는 이미지)
```bash
python cybercop_pipeline_AdotX.py --file "data/7.png"
```

### CSV 배치
```bash
python cybercop_pipeline_AdotX.py --csv "data/labels.csv"
```

### 로컬 디렉토리 (영상/이미지 파일 일괄 처리)
```bash
python cybercop_pipeline_AdotX.py --dir "C:\사이버 범죄 데이터\직거래 사기"
```

> 디렉토리 내 지원 확장자 파일을 자동으로 탐색하여 순차 처리합니다.
> 공백이나 한글이 포함된 경로는 반드시 **따옴표로 감싸서** 입력하세요.
>
> 지원 영상 확장자: `.mp4` `.avi` `.mkv` `.mov` `.webm` `.flv` `.ts`
> 지원 이미지 확장자: `.jpg` `.jpeg` `.png` `.bmp` `.gif` `.webp` `.tiff`

### 옵션
| 인자 | 기본값 | 설명 |
|---|---|---|
| `--out_dir` | `./downloaded_videos` | 결과 저장 경로 |
| `--sample_sec` | `2.0` | 프레임 샘플링 간격 (초) |
| `--max_frames` | `10` | 영상당 최대 프레임 수 |
| `--top_k` | `3` | RAG 검색 상위 문서 수 |
| `--model` | `skt/A.X-4.0-VL-Light` | HuggingFace 모델 ID |

---

## 처리 파이프라인

```
프레임 추출 → VLM 추론(OCR + 객체) → OCR 병합 → VLM OCR 후처리 → RAG 검색 → 레이블 자동 예측
```

1. **프레임 추출**: 영상에서 `sample_sec` 간격으로 최대 `max_frames`개 샘플링
2. **VLM 추론**: 각 프레임에서 OCR 텍스트와 객체 목록 추출
3. **OCR 병합**: 연속 프레임 간 유사도(≥0.5)로 중복 제거, 고유 줄만 누적
4. **VLM OCR 후처리**: 병합된 OCR을 VLM에 재입력하여 오타·할루시네이션 교정 및 문장 연결 (@아이디, URL, 전화번호 등 식별자는 원문 보존)
5. **RAG 검색**: BGE-m3 임베딩으로 범죄 유형 문서 검색 (FAISS)
6. **레이블 예측**: RAG 최고 유사도 ≥ 0.5 → `abnormal`, 미만 → `normal` (rag 결과 없음)

---

## 객체 인식 방식

VLM이 자유롭게 출력한 객체명을 **임베딩 유사도(BGE-m3)**로 `rag/allowed_objects.json`에 정의된 허용 클래스(446종)에 매핑합니다.
프롬프트에 전체 클래스 목록을 포함하지 않아 inference 속도가 빠르며, 코드 수정 없이 JSON 파일만 편집해 클래스 목록을 관리할 수 있습니다.

---

## 허용 객체 DB 관리 (`review_and_add_objects.py`)

새로운 영상에서 추출한 객체를 검토하여 `rag/allowed_objects.json`과 `rag/retrieval_docs.json`에 일괄 추가합니다.

```bash
# 분석·미리보기만 (파일 수정 없음)
python review_and_add_objects.py --dry_run

# 실제 추가 (기본값: 중복 임계값 0.88, 범죄 관련성 임계값 0.55, 최소 영상 2개)
python review_and_add_objects.py

# 파라미터 조정
python review_and_add_objects.py --sim_thresh 0.85 --crime_thresh 0.4 --min_videos 2

# RAG DB 추가 건너뜀 (allowed_objects.json만 업데이트)
python review_and_add_objects.py --no_rag
```

입력 파일: `rag/new_cyber_objects.json` (extract_cyber_objects.py 실행 결과)

---

## Input / Output 정의

### Input
| 항목 | 형식 | 설명 |
|---|---|---|
| `--url` | URL 문자열 | YouTube / TikTok 단일 영상 |
| `--file` | 파일 경로 | 로컬 영상 또는 이미지 단일 파일 |
| `--csv` | CSV 파일 경로 | `label`, `link` 컬럼 포함 |
| `--dir` | 디렉토리 경로 | 영상/이미지 파일이 담긴 로컬 폴더 |

CSV 형식:
```csv
label,link
abnormal,https://www.youtube.com/shorts/xxxxx
normal,https://www.tiktok.com/@user/video/xxxxx
```

### Output
영상/이미지 1개당 `downloaded_videos/results/`에 2개 파일 저장:

**`ocr_results_{id}.json`** — UI 연동용
```json
{
  "id": "영상ID",
  "title": "영상 제목",
  "gt_label": "abnormal",
  "label": "abnormal",
  "objects": ["휴대전화", "사람"],
  "ocr_frames": [
    { "ts": "00:00", "text": "지금 투자하면 300% 수익 보장!" },
    { "ts": "00:02", "text": "지금 투자하면 300% 수익 보장!\n@kakao_id" }
  ],
  "ocr_merged": [
    {
      "start": "00:00",
      "end": "00:04",
      "text": "지금 투자하면 300% 수익 보장!\n@kakao_id"
    }
  ],
  "rag": [
    {
      "crime_type": "피싱",
      "text": "피싱: 개인정보 탈취를 위한 사기",
      "similarity": 0.87,
      "risk_level": 0.22
    }
  ]
}
```

**`ocr_merged_{id}.txt`** — 사람이 읽기 쉬운 요약

### Output 필드
| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | string | 영상/파일 고유 ID |
| `title` | string | 영상 제목 또는 파일명 |
| `gt_label` | string | 입력 레이블 (CSV의 label 또는 "manual") |
| `label` | string | RAG 유사도 기반 자동 예측 레이블 (abnormal / normal) |
| `objects` | string[] | 감지된 객체 목록 (한국어, 2프레임 이상 등장) |
| `ocr_frames[].ts` | string | 프레임 타임스탬프 |
| `ocr_frames[].text` | string | 해당 프레임의 원시 OCR 텍스트 |
| `ocr_merged[].start` | string | 병합 구간 시작 타임스탬프 |
| `ocr_merged[].end` | string | 병합 구간 종료 타임스탬프 |
| `ocr_merged[].text` | string | VLM 후처리된 최종 OCR 텍스트 |
| `rag[].crime_type` | string | 매칭된 범죄 유형 |
| `rag[].similarity` | float | 코사인 유사도 (0~1) |
| `rag[].risk_level` | float | 위험도 (0~1, 1이 최고) |

> `label`이 `normal`인 경우 `rag` 필드는 빈 배열(`[]`)로 반환됩니다.

### 지원 범죄 유형 (26종)
직거래 사기, 쇼핑몰 사기, 게임 사기, 이메일 무역사기, 기타 사이버 사기,
피싱, 파밍, 스미싱, 메모리 해킹, 몸캠피싱, 메신저 이용사기, 기타 사이버 금융범죄,
개인·위치정보 침해, 사이버저작권 침해, 기타 정보통신망 이용범죄,
아동성 착취물, 불법 촬영물, 허위 영상물, 불법성 영상물, 기타 불법콘텐츠 범죄,
스포츠 토토, 경마·경륜·경정, 카지노, 기타 사이버 도박, 명예훼손, 모욕

## API 서버

CLI 스크립트를 FastAPI 서버로 래핑한 `app/main.py`를 제공합니다.

### 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

서버 시작 시 VLM 모델(A.X-4.0-VL-Light)과 RAG 인덱스를 자동으로 로드합니다.

### 엔드포인트

#### `POST /api/video` — YouTube / TikTok URL 분석

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `url` | string (Form) | 필수 | YouTube / TikTok 영상 URL |
| `sample_sec` | float (Form) | `1.0` | 프레임 샘플링 간격 (초) |
| `max_frames` | int (Form) | `10` | 최대 프레임 수 |
| `top_k` | int (Form) | `3` | RAG 검색 상위 k |

```bash
curl -X POST http://localhost:8000/api/video \
  -F "url=https://www.youtube.com/shorts/영상ID"
```

---

#### `POST /api/video/upload` — 영상 / 이미지 파일 업로드 분석

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `file` | UploadFile | 필수 | 영상(.mp4 등) 또는 이미지(.jpg 등) 파일 |
| `sample_sec` | float (Form) | `1.0` | 프레임 샘플링 간격 (초) |
| `max_frames` | int (Form) | `10` | 최대 프레임 수 |
| `top_k` | int (Form) | `3` | RAG 검색 상위 k |

```bash
curl -X POST http://localhost:8000/api/video/upload \
  -F "file=@영상파일.mp4"
```

---

#### `POST /api/video/csv` — CSV 배치 분석

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `file` | UploadFile | 필수 | `link`, `label` 컬럼을 가진 CSV 파일 |
| `sample_sec` | float (Form) | `1.0` | 프레임 샘플링 간격 (초) |
| `max_frames` | int (Form) | `10` | 최대 프레임 수 |
| `top_k` | int (Form) | `3` | RAG 검색 상위 k |

```bash
curl -X POST http://localhost:8000/api/video/csv \
  -F "file=@data/labels.csv"
```

---

### API 응답 형식

**단건 (`/api/video`, `/api/video/upload`)**
```json
{
  "message": "success",
  "result": {
    "id": "영상ID",
    "title": "영상 제목",
    "label": "abnormal",
    "objects": ["휴대전화", "채팅창", "사람"],
    "ocr_frames": [
      { "ts": "00:01", "text": "지금 투자하면 300% 수익 보장!" }
    ],
    "ocr_merged": [
      { "start": "00:01", "end": "00:03", "text": "지금 투자하면 300% 수익 보장!" }
    ],
    "rag": [
      { "crime_type": "피싱", "text": "피싱: 개인정보 탈취를 위한 사기", "similarity": 0.87 }
    ]
  }
}
```

**배치 (`/api/video/csv`)**
```json
{
  "message": "success",
  "count": 2,
  "results": [
    {
      "id": "...", "title": "...", "gt_label": "abnormal", "label": "abnormal",
      "objects": [], "ocr_frames": [], "ocr_merged": [], "rag": [], "error": null
    },
    { "url": "...", "label": "...", "error": "실패 사유" }
  ]
}
```

### API 응답 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | string | 영상 고유 ID |
| `title` | string | 영상 제목 |
| `label` | string | RAG 유사도 기반 자동 예측 (abnormal / normal) |
| `gt_label` | string | CSV 입력 레이블 (배치 모드만) |
| `objects` | string[] | 감지된 객체 목록 (한국어) |
| `ocr_frames[].ts` | string | 프레임 타임스탬프 |
| `ocr_frames[].text` | string | 프레임 원시 OCR 텍스트 |
| `ocr_merged[].start` | string | 병합 구간 시작 타임스탬프 |
| `ocr_merged[].end` | string | 병합 구간 종료 타임스탬프 |
| `ocr_merged[].text` | string | VLM 후처리된 최종 OCR 텍스트 |
| `rag[].crime_type` | string | 매칭된 범죄 유형 |
| `rag[].similarity` | float | 코사인 유사도 (0~1) |

> `label`이 `normal`인 경우 `rag` 필드는 빈 배열(`[]`)로 반환됩니다.
