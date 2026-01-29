## QA 디바이스 추천 자동화 (devices_auto)

Jira Bug 이력을 기반으로 QA 테스트 디바이스 후보를 **추천(결정 아님)** 하고, 각 추천에 대해 **템플릿 기반 Why(추천 사유)** 를 제공합니다.

### 핵심 개념
- **Spec Cluster**: `AP(SoC Family) × RAM Tier × (옵션) Resolution Class`
- **RAM Tier**
  - Low: ≤ 3GB
  - Mid: 4~6GB
  - High: ≥ 8GB
- **Slot 구조**
  - **Slot A**: 프로젝트 요청/정책 필수
  - **Slot B**: 리스크 상위 Spec Cluster 대표(또는 미검증 디바이스 우선)
  - **Slot C**: 미검증 공백(항상 1개 이상)

### 설치

```bash
cd devices_auto
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 실행 (PoC)

```bash
#
# 현재 버전은 `src.devices_auto.cli` 엔트리포인트가 아니라 `scripts/`의 실행 스크립트를 사용합니다.
# (README의 이전 CLI 예시는 구버전/스캐폴딩 기준이라 실제 코드와 다릅니다.)
#

# 0) (권장) 환경변수 설정
# - `env.example`를 참고해서 `.env`를 만들어 값만 채우세요. (`.env`는 gitignore)

# 1) Jira 버그 데이터 export (raw json + flat csv)
python scripts/fetch_jira_bugs.py --jql "$env:JIRA_JQL" --overwrite

# 2) (선택) Testbed 정규화
# python scripts/normalize_testbed.py --input data/testbed.xlsx --out-dir data

# 3) Jira <-> Testbed 매칭/보강(enriched) 생성
# python scripts/enrich_jira_with_testbed.py --out-dir data

# 4) Spec Cluster 리스크 리포트 생성
# python scripts/cluster_risk_report.py --project-key PGPAM --out-dir data

# 5) 후보 풀 생성(eligible_candidates.csv 단일 파일)
# python scripts/eligible_candidates.py --project-key PGPAM --out-dir data

# 6) Slot A/B/C 추천 생성
# python scripts/recommend_slots.py --project-key PGPAM --out-dir data
```

### 입력/출력 파일 (현행 스크립트 기준)
- **입력(대표)**:
  - `data/testbed_normalized*.csv` (testbed 정규화 결과)
  - `data/jira_bugs_flat*.csv` 또는 `data/jira_bugs_enriched_*.csv` (Jira export/보강 결과)
  - `data/cluster_risk_{PROJECT}_*.csv` (Spec Cluster 리스크)
  - `data/eligible_candidates.csv` (추천 후보 풀)
- **출력(대표)**:
  - `data/recommendation_final_{PROJECT}.csv` (최종 추천)
  - `data/recommendation_report_{PROJECT}.csv` (추천 사유/로그용)

### (참고) 구버전 입력 포맷
아래 포맷은 과거 CLI/실험용 문서에 남아있는 스펙입니다. 현행 추천 플로우는 `scripts/` 파이프라인을 사용합니다.
- `data/device_catalog.csv`
  - 필수: `device_name,ap_family,ram_gb,resolution_w,resolution_h`
  - 선택: `market_share_weight`
- `data/jira_bugs.csv`
  - 필수: `issue_key,fix_version,severity,reopened`
  - 선택: `device_name,ap_family,ram_gb,created`

### FastAPI (Confluence Constraint Scan API)
`app/main.py`에 FastAPI 서버가 있습니다. (v1은 Confluence 크롤링 없이 `input_texts`만 처리)

```bash
uvicorn app.main:app --reload --port 8000
```

- Job 생성: `POST /api/confluence/constraint-scan/jobs`
- 상태 조회: `GET /api/confluence/constraint-scan/jobs/{job_id}`
- 결과 조회: `GET /api/confluence/constraint-scan/jobs/{job_id}/result?format=html`


