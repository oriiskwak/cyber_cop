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
│   └── retrieval_docs.json        # 범죄 유형 문서 (26종)
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

## 객체 인식 방식

VLM이 자유롭게 출력한 객체명을 **임베딩 유사도(BGE-m3)**로 Objects365 한국어 클래스(200종)에 매핑합니다.
프롬프트에 전체 클래스 목록을 포함하지 않아 inference 속도가 빠릅니다.

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
  "label": "abnormal",
  "objects": ["휴대전화", "사람"],
  "ocr": [
    {
      "start": "00:00",
      "end": "00:02",
      "text": "지금 투자하면 300% 수익 보장!"
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
| `label` | string | 입력 레이블 (abnormal / normal / manual) |
| `objects` | string[] | 감지된 객체 목록 (한국어) |
| `ocr[].start` | string | OCR 구간 시작 타임스탬프 |
| `ocr[].end` | string | OCR 구간 종료 타임스탬프 |
| `ocr[].text` | string | 추출된 OCR 텍스트 |
| `rag[].crime_type` | string | 매칭된 범죄 유형 |
| `rag[].similarity` | float | 코사인 유사도 (0~1) |
| `rag[].risk_level` | float | 위험도 (0~1, 1이 최고) |

### 지원 범죄 유형 (26종)
직거래 사기, 쇼핑몰 사기, 게임 사기, 이메일 무역사기, 기타 사이버 사기,
피싱, 파밍, 스미싱, 메모리 해킹, 몸캠피싱, 메신저 이용사기, 기타 사이버 금융범죄,
개인·위치정보 침해, 사이버저작권 침해, 기타 정보통신망 이용범죄,
아동성 착취물, 불법 촬영물, 허위 영상물, 불법성 영상물, 기타 불법콘텐츠 범죄,
스포츠 토토, 경마·경륜·경정, 카지노, 기타 사이버 도박, 명예훼손, 모욕
