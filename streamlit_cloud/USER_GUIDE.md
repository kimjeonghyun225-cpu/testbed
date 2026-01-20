### QA 디바이스 추천 자동화 (testbed_auto / Streamlit Cloud) — 사용자 설명서

이 문서는 Streamlit Cloud 배포 폴더(`streamlit_cloud`) 기준으로 작성되었습니다.

---

## API 연결 설정(세션/저장)

### 세션 적용
- `세션 적용` 버튼을 누르면 입력한 키가 **현재 실행 중인 Streamlit 세션**에 반영됩니다.
- 반영 후 OpenAI healthcheck를 수행해 `정상/비정상(401 등)` 상태를 표시합니다.

### local.env에 저장 / local.env 다시 읽기
- 로컬(내 PC)에서는 `local.env` 파일을 저장/로드하는 방식이 유용합니다.
- 하지만 Streamlit Cloud(배포)에서는 컨테이너 파일시스템 특성상 **`local.env` 저장/유지가 보장되지 않습니다.**
- 따라서 배포에서는 아래의 **Secrets 방식**을 권장합니다.

---

## 배포 환경 권장 방식: Streamlit Secrets

1. 앱 화면 우측 하단 **Manage app**
2. **Settings → Secrets**에 아래처럼 등록

```toml
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4.1-mini"
JIRA_BASE_URL="https://xxx.atlassian.net"
JIRA_EMAIL="your@email.com"
JIRA_API_TOKEN="..."
```

3. 앱 재실행 후 즉시 적용됩니다.

---

## 정책(YAML) 기반 결과 추출 동작 원리(요약)

1. 정책 선택(KP/KRJP/PALM) → YAML 로딩
2. 마스터 디바이스 업로드 → 헤더/시트 탐지 → 표준 컬럼 정규화
3. candidate_filter로 후보 필터링(platform/usable_only/required_fields/범위 필터 등)
4. Rank별 목표 수량을 채우며 대표성/다양성/타이브레이커 적용
5. 결과 표 표시 및 엑셀 다운로드

