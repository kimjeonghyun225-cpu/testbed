from __future__ import annotations

import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_V2_DIR = PROJECT_ROOT / "config" / "policies_v2"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"


def _ensure_repo_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def _safe_str(v: Any) -> str:
    s = str(v or "").strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def _split_csv_list(s: str) -> list[str]:
    if not _safe_str(s):
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


@st.cache_data(show_spinner=False)
def _extract_target_country_options_from_upload(file_name: str, data: bytes) -> list[str]:
    """
    업로드 파일에서 '타겟 국가' 후보 옵션을 best-effort로 추출한다.
    (정규화 전체를 돌리지 않고, 해당 컬럼만 찾아 unique 정렬)
    """
    name = (file_name or "").lower()

    def _norm_col(x: object) -> str:
        s = str(x or "").strip().lower()
        for ch in [" ", "\t", "\n", "\r", "-", "_", ".", "/", "\\", "(", ")", "[", "]", "{", "}", ":", ";", "|"]:
            s = s.replace(ch, "")
        return s

    def _pick_target_col(columns: list[object]) -> str | None:
        aliases = ["타겟국가", "targetmarket", "targetcountry", "country", "market", "region"]
        best: str | None = None
        best_score = 0
        for c in columns:
            n = _norm_col(c)
            score = 0
            if "타겟" in n and "국가" in n:
                score += 100
            if "target" in n and ("country" in n or "market" in n):
                score += 80
            for a in aliases:
                if a in n:
                    score += 10
            if score > best_score:
                best_score = score
                best = str(c)
        return best if best_score > 0 else None

    def _read_excel_sheet_with_header_guess(xbytes: bytes, sheet_name: str | int) -> pd.DataFrame | None:
        import io as _io

        # header 없이 상단 일부를 읽어서 "헤더 행"을 찾는다 (L6 같은 케이스 대응)
        preview = pd.read_excel(_io.BytesIO(xbytes), sheet_name=sheet_name, header=None, nrows=40)
        header_row: int | None = None
        for i in range(len(preview)):
            row_vals = [
                str(x)
                for x in preview.iloc[i].tolist()
                if str(x).strip() and str(x).strip().lower() != "nan"
            ]
            if not row_vals:
                continue
            if _pick_target_col(row_vals) is not None:
                header_row = i
                break

        if header_row is None:
            # fallback: 기본 header=0
            return pd.read_excel(_io.BytesIO(xbytes), sheet_name=sheet_name)
        return pd.read_excel(_io.BytesIO(xbytes), sheet_name=sheet_name, header=header_row)

    try:
        import io as _io

        if name.endswith(".csv"):
            df = pd.read_csv(_io.BytesIO(data))
        else:
            df = None
            # Device_Info 시트 우선 (대소문자 무시)
            try:
                xls = pd.ExcelFile(_io.BytesIO(data))
                sheets = list(xls.sheet_names)
            except Exception:
                sheets = []

            preferred = None
            for sname in sheets:
                if str(sname).strip().lower() == "device_info":
                    preferred = sname
                    break

            if preferred is not None:
                try:
                    df = _read_excel_sheet_with_header_guess(data, preferred)
                except Exception:
                    df = None

            # fallback: 첫 시트
            if df is None or df.empty:
                try:
                    df = _read_excel_sheet_with_header_guess(data, 0)
                except Exception:
                    df = None
    except Exception:
        return []

    if df is None or df.empty:
        return []

    target_col_name = _pick_target_col(list(df.columns))
    if target_col_name is None or target_col_name not in df.columns:
        return []

    s = df[target_col_name].fillna("").astype(str).str.strip()
    opts = sorted({x for x in s.tolist() if x and x.lower() not in ("nan", "none", "null")})
    return opts


@st.cache_data(show_spinner=False)
def _extract_note_options_from_upload(file_name: str, data: bytes) -> list[str]:
    """
    업로드 파일에서 'NOTE/비고' 후보 옵션을 best-effort로 추출한다.
    (정규화 전체를 돌리지 않고, 해당 컬럼만 찾아 unique 정렬)
    """
    name = (file_name or "").lower()

    def _norm_col(x: object) -> str:
        s = str(x or "").strip().lower()
        for ch in [" ", "\t", "\n", "\r", "-", "_", ".", "/", "\\", "(", ")", "[", "]", "{", "}", ":", ";", "|"]:
            s = s.replace(ch, "")
        return s

    def _pick_note_col(columns: list[object]) -> str | None:
        aliases = ["note", "비고", "remarks", "remark", "memo", "comment", "comments"]
        best: str | None = None
        best_score = 0
        for c in columns:
            n = _norm_col(c)
            score = 0
            if n == "note":
                score += 120
            if "note" in n:
                score += 80
            if "비고" in n:
                score += 80
            for a in aliases:
                if a in n:
                    score += 10
            if score > best_score:
                best_score = score
                best = str(c)
        return best if best_score > 0 else None

    def _read_excel_sheet_with_header_guess(xbytes: bytes, sheet_name: str | int) -> pd.DataFrame | None:
        import io as _io

        preview = pd.read_excel(_io.BytesIO(xbytes), sheet_name=sheet_name, header=None, nrows=40)
        header_row: int | None = None
        for i in range(len(preview)):
            row_vals = [
                str(x)
                for x in preview.iloc[i].tolist()
                if str(x).strip() and str(x).strip().lower() != "nan"
            ]
            if not row_vals:
                continue
            if _pick_note_col(row_vals) is not None:
                header_row = i
                break

        if header_row is None:
            return pd.read_excel(_io.BytesIO(xbytes), sheet_name=sheet_name)
        return pd.read_excel(_io.BytesIO(xbytes), sheet_name=sheet_name, header=header_row)

    try:
        import io as _io

        if name.endswith(".csv"):
            df = pd.read_csv(_io.BytesIO(data))
        else:
            df = None
            try:
                xls = pd.ExcelFile(_io.BytesIO(data))
                sheets = list(xls.sheet_names)
            except Exception:
                sheets = []

            preferred = None
            for sname in sheets:
                if str(sname).strip().lower() == "device_info":
                    preferred = sname
                    break

            if preferred is not None:
                try:
                    df = _read_excel_sheet_with_header_guess(data, preferred)
                except Exception:
                    df = None

            if df is None or df.empty:
                try:
                    df = _read_excel_sheet_with_header_guess(data, 0)
                except Exception:
                    df = None
    except Exception:
        return []

    if df is None or df.empty:
        return []

    note_col_name = _pick_note_col(list(df.columns))
    if note_col_name is None or note_col_name not in df.columns:
        return []

    s = df[note_col_name].fillna("").astype(str).str.strip()
    opts = sorted({x for x in s.tolist() if x and x.lower() not in ("nan", "none", "null")})
    return opts


def _parse_platforms(platform_txt: str) -> list[str]:
    s = _safe_str(platform_txt).lower()
    if not s:
        return ["android"]
    # allow "android,ios" / "android ios" / "android|ios"
    for ch in ["|", ";"]:
        s = s.replace(ch, ",")
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p in ("aos", "android"):
            out.append("android")
        elif p in ("ios", "iphone", "ipad"):
            out.append("ios")
        elif p in ("both", "all", "*"):
            return ["android", "ios"]
    # unique preserve order
    seen = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq or ["android"]


def _save_upload(upload, *, prefix: str) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_str(getattr(upload, "name", "")) or f"{prefix}.bin"
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-", ".", " ")).strip()
    if not safe:
        safe = f"{prefix}.bin"
    out = UPLOADS_DIR / f"{prefix}__{safe}"
    out.write_bytes(upload.getbuffer())
    return out


def _dedupe_cols(xdf: pd.DataFrame) -> pd.DataFrame:
    if xdf is None or xdf.empty:
        return xdf
    out = xdf.copy()
    counts: dict[str, int] = {}
    cols: list[str] = []
    for c in list(out.columns):
        base = str(c)
        n = counts.get(base, 0) + 1
        counts[base] = n
        cols.append(base if n == 1 else f"{base}({n})")
    out.columns = cols
    return out


def _reorder_preferred_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    중복 제거 전에 '유지하고 싶은 표준 컬럼'을 앞으로 당겨서,
    동일 값 컬럼이 여러 개일 때 표준 컬럼이 남도록 한다.
    """
    if df is None or df.empty:
        return df
    preferred = [
        # identity / core
        "No",
        "device_id",
        "product_name",
        "model_name",
        "model_number",
        "brand",
        "rank",
        "available",
        # specs
        "cpu",
        "cpu_family",
        "gpu",
        "ram_gb",
        "display_w",
        "display_h",
        "display_bucket",
        "form_factor",
        "target_market",
        "release_year",
        # engine explainability
        "profile_core",
        "axes_key",
    ]
    cols = list(df.columns)
    head = [c for c in preferred if c in cols]
    tail = [c for c in cols if c not in head]
    return df[head + tail].copy()


def _drop_duplicate_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    값이 완전히 동일한(전체 행 기준) 컬럼은 하나만 남기고 제거한다.
    - 성능을 위해 각 컬럼을 문자열로 정규화 후 hash signature로 비교
    """
    if df is None or df.empty:
        return df
    tmp = df.copy()
    # normalize to strings (stable comparison)
    sig_to_col: dict[int, str] = {}
    keep: list[str] = []
    for c in list(tmp.columns):
        s = tmp[c]
        try:
            norm = s.fillna("").astype(str).str.strip()
        except Exception:
            norm = s.astype(str)
        # include column name in tie-breaker? no — value-only dedupe
        sig = int(pd.util.hash_pandas_object(norm, index=False).sum())
        if sig in sig_to_col:
            # potential collision extremely unlikely; verify equality
            prev = sig_to_col[sig]
            try:
                if norm.equals(tmp[prev].fillna("").astype(str).str.strip()):
                    continue
            except Exception:
                # if comparison fails, keep it
                pass
        sig_to_col[sig] = c
        keep.append(c)
    return tmp[keep].copy()


def _clean_result_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    결과 화면/엑셀용 DF 정리:
    - 표준 컬럼을 앞에 배치
    - 값이 같은 중복 컬럼 제거
    """
    if df is None or df.empty:
        return df

    # 1) 사용자 출력 스키마(원본 엑셀 헤더 기반)로 재구성
    out = _to_user_output_schema(df)
    # 2) 혹시 남아있을 수 있는 값-중복 컬럼 제거(안전)
    out = _drop_duplicate_value_columns(out)
    return out


def _to_user_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    사용자가 원하는 결과 컬럼/순서로 재구성한다.
    - 가능하면 원본 컬럼(엑셀 헤더)을 우선 사용
    - 없으면 v2 표준 컬럼에서 파생
    """
    if df is None or df.empty:
        return df

    # 원하는 출력 헤더(순서)
    # 결과 표/엑셀에서 Rank를 맨 앞에 두고 싶다는 요구 반영
    out_cols = [
        "Rank",
        "No",
        "자산번호",
        "상태",
        "대여가능여부",
        "OS",
        "제품명",
        "모델번호",
        "제조사",
        "Rating",
        "디바이스 타입",
        "타겟 국가",
        "CPU",
        "GPU",
        "RAM",
        "DISPLAY",
        "출시 년도",
        "아키텍처",
        "OS Ver",
        "선정 사유",
    ]

    # 각 출력 헤더에 대응되는 후보 컬럼들(앞의 것이 우선)
    mapping: dict[str, list[str]] = {
        # "No"는 프로젝트/엑셀마다 표기가 흔들림: NO, No., 번호 등도 허용
        "No": ["No", "NO", "no", "No.", "NO.", "번호", "관리번호", "자산 No", "Asset No", "AssetNo"],
        "자산번호": ["자산번호", "device_id", "raw__자산번호"],
        "상태": ["상태", "raw__상태", "status"],
        "대여가능여부": ["대여가능여부", "raw__대여가능여부", "rentable"],
        "OS": ["OS", "os"],
        "제품명": ["제품명", "product_name", "model_name"],
        "모델번호": ["모델번호", "model_number"],
        "제조사": ["제조사", "brand"],
        "Rating": ["Rating", "rating"],
        "Rank": ["Rank", "rank"],
        "디바이스 타입": ["디바이스 타입", "form_factor"],
        "타겟 국가": ["타겟 국가", "target_market"],
        "CPU": ["CPU", "cpu", "ap_family"],
        "GPU": ["GPU", "gpu"],
        "RAM": ["RAM", "ram_gb"],
        "DISPLAY": ["DISPLAY", "display", "해상도"],
        "출시 년도": ["출시 년도", "출시년도", "release_year"],
        "아키텍처": ["아키텍처", "arch"],
        "OS Ver": ["OS Ver", "OSVer", "os_ver"],
        "선정 사유": ["why_detail", "why"],
    }

    def _norm_key(x: object) -> str:
        # remove ALL whitespace (including NBSP) + common punctuations + BOM
        s = str(x or "").replace("\ufeff", "").strip().lower()
        s = re.sub(r"\s+", "", s)
        for ch in ["-", "_", ".", "(", ")", "[", "]", "{", "}", ":", ";", "/", "\\", "|"]:
            s = s.replace(ch, "")
        return s

    # normalized lookup for header variations (e.g., "No " / "NO." / hidden spaces)
    # NOTE: 같은 norm key를 가진 컬럼이 여러 개일 수 있어(빈 "No" + 실제 값 있는 "No " 등)
    # 가장 값이 많이 채워진 컬럼을 선택한다.
    norm_to_cols: dict[str, list[str]] = {}
    for c in df.columns:
        k = _norm_key(c)
        if not k:
            continue
        norm_to_cols.setdefault(k, []).append(str(c))

    def _best_non_empty(cols: list[str]) -> str | None:
        best: str | None = None
        best_cnt = -1
        for c in cols:
            if c not in df.columns:
                continue
            try:
                s = df[c].fillna("").astype(str).str.strip()
                cnt = int(s.ne("").sum())
            except Exception:
                cnt = 0
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
        return best if best_cnt > 0 else (best or None)

    def _first_existing(cols: list[str]) -> str | None:
        # 1) exact match first
        for c in cols:
            if c in df.columns:
                return c
        # 2) normalized match (ignore spaces/punctuations/case)
        for c in cols:
            k = _norm_key(c)
            got_list = norm_to_cols.get(k) or []
            got = _best_non_empty(got_list) if got_list else None
            if got and got in df.columns:
                return got
        return None

    out = pd.DataFrame(index=df.index)
    for out_name in out_cols:
        src = _first_existing(mapping.get(out_name, []))
        # "No"는 파일마다 비어있거나 컬럼명이 흔들리는 경우가 많아,
        # 사진 기준의 규칙(AP/IP/AT/IT/R로 시작)을 만족하는 값을 가장 많이 포함하는 컬럼을 우선 선택한다.
        if out_name == "No":
            NO_CODE_RE = re.compile(r"(?i)^(AP|IP|AT|IT|R)\\d+")
            cand_cols = mapping.get("No", [])
            # add any column that normalizes to 'no' (e.g., 'No ' / 'NO.' / hidden chars)
            for c in df.columns:
                if _norm_key(c) == "no" and str(c) not in cand_cols:
                    cand_cols = [str(c)] + list(cand_cols)

            best_src = None
            best_score = (-1, -1)  # (pattern_match_count, non_empty_count)
            for c in cand_cols:
                cc = _first_existing([c])
                if not cc:
                    continue
                s = df[cc].fillna("").astype(str).str.strip()
                non_empty = int(s.ne("").sum())
                pat = int(s.map(lambda x: bool(NO_CODE_RE.match(str(x)))).sum())
                score = (pat, non_empty)
                if score > best_score:
                    best_score = score
                    best_src = cc

            if best_src and best_score[1] > 0:
                out[out_name] = df[best_src].fillna("").astype(str).str.strip()
            else:
                # fallback: 임시 No (원본 No를 못 찾았을 때 혼동 방지)
                out[out_name] = pd.Series(range(1, len(df) + 1), index=df.index).apply(lambda x: f"TMP{x:04d}").astype(str)
            continue

        if out_name == "RAM":
            if src and src != "ram_gb":
                out[out_name] = df[src]
            elif "ram_gb" in df.columns:
                rg = pd.to_numeric(df["ram_gb"], errors="coerce")
                out[out_name] = rg.apply(lambda x: "" if pd.isna(x) else f"{int(x)}GB")
            else:
                out[out_name] = ""
            continue
        if out_name == "DISPLAY":
            # prefer existing display string, else derive from display_w/h
            if src:
                out[out_name] = df[src]
            elif "display_w" in df.columns and "display_h" in df.columns:
                w = pd.to_numeric(df["display_w"], errors="coerce")
                h = pd.to_numeric(df["display_h"], errors="coerce")
                out[out_name] = pd.concat([w, h], axis=1).apply(
                    lambda r: "" if pd.isna(r[0]) or pd.isna(r[1]) else f"{int(r[0])} x {int(r[1])}",
                    axis=1,
                )
            else:
                out[out_name] = ""
            continue

        if src:
            out[out_name] = df[src]
        else:
            out[out_name] = ""

    # strip NaN-like to empty for readability
    for c in out.columns:
        # also cover numpy.nan
        try:
            out[c] = out[c].fillna("")
        except Exception:
            out[c] = out[c].replace({pd.NA: "", None: ""})
    return out


def _attach_why(selected_df: pd.DataFrame, decision_log: pd.DataFrame) -> pd.DataFrame:
    """
    선택 결과 DF에 선정 사유(why/why_detail)를 조인한다.
    키 우선순위: device_id -> No
    """
    if selected_df is None or selected_df.empty or decision_log is None or decision_log.empty:
        return selected_df
    sel = selected_df.copy()
    dl = decision_log.copy()
    for c in ["device_id", "No"]:
        if c in sel.columns:
            sel[c] = sel[c].fillna("").astype(str).str.strip()
        if c in dl.columns:
            dl[c] = dl[c].fillna("").astype(str).str.strip()

    # IMPORTANT:
    # - merge on device_id is preferred, but decision_log also has "No" which would create No_x/No_y
    #   and later UI might not find "No". So when merging on device_id, we intentionally DO NOT
    #   bring "No" from decision_log.
    if "device_id" in sel.columns and "device_id" in dl.columns and sel["device_id"].fillna("").astype(str).str.len().gt(0).any():
        cols_keep = [c for c in ["device_id", "why", "why_detail"] if c in dl.columns]
        if not cols_keep:
            return sel
        dl2 = dl[cols_keep].copy()
        out = sel.merge(dl2.drop_duplicates(subset=["device_id"]), on="device_id", how="left")
    elif "No" in sel.columns and "No" in dl.columns:
        cols_keep = [c for c in ["No", "why", "why_detail"] if c in dl.columns]
        if not cols_keep:
            return sel
        dl2 = dl[cols_keep].copy()
        out = sel.merge(dl2.drop_duplicates(subset=["No"]), on="No", how="left")
    else:
        return sel

    # Safety: if No got suffixed by any chance, restore canonical "No"
    if "No" not in out.columns:
        if "No_x" in out.columns:
            out["No"] = out["No_x"]
        elif "No_y" in out.columns:
            out["No"] = out["No_y"]
    out = out.drop(columns=["No_x", "No_y"], errors="ignore")
    return out


def _read_env_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip()
        if val and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        out[key] = val
    return out


def _write_env_kv(path: Path, updates: dict[str, str]) -> None:
    cur = _read_env_kv(path)
    for k, v in updates.items():
        sv = str(v or "").strip()
        if sv:
            cur[str(k).strip()] = sv
    lines = [f'{k}="{cur[k]}"' for k in sorted(cur.keys())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _openai_chat(*, api_key: str, model: str, user_text: str, system_text: str) -> str:
    """
    Minimal OpenAI ChatCompletions wrapper via REST (no openai package).
    """
    api_key = _safe_str(api_key)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 비어 있습니다. (API 연결 설정에서 세션 적용/저장 후 다시 시도)")
    model = _safe_str(model) or "gpt-4.1-mini"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI 요청 실패: {r.status_code} {r.text[:500]}")
    data = r.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def _openai_healthcheck(api_key: str) -> tuple[bool, str]:
    """
    OpenAI 키가 유효한지 best-effort로 확인한다.
    """
    api_key = _safe_str(api_key)
    if not api_key:
        return (False, "OPENAI_API_KEY 없음")
    try:
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if r.status_code >= 400:
            return (False, f"OpenAI 인증 실패: {r.status_code}")
        return (True, "OpenAI OK")
    except Exception as e:
        return (False, f"OpenAI 확인 실패: {e}")


def _policy_local_summary(yaml_text: str) -> str:
    """
    OpenAI 없이도 볼 수 있는 정책 요약(로컬).
    """
    try:
        obj = yaml.safe_load(yaml_text or "")
    except Exception as e:
        return f"YAML 파싱 실패: {e}"
    if not isinstance(obj, dict):
        return "정책 YAML 최상위가 dict가 아닙니다."
    proj = str(obj.get("project") or "")
    cf = obj.get("candidate_filter") or {}
    dedupe = obj.get("dedupe") or {}
    diversity = obj.get("diversity") or {}
    manu = obj.get("manufacturer_policy") or {}
    tb = obj.get("tie_breakers") or []
    lines = []
    lines.append(f"- project: {proj}")
    lines.append(f"- candidate_filter: {cf}")
    lines.append(f"- dedupe: {dedupe}")
    lines.append(f"- diversity: {diversity}")
    lines.append(f"- manufacturer_policy: {manu}")
    lines.append(f"- tie_breakers: {tb}")
    return "\n".join(lines)


def _safe_policy_filename(stem: str) -> str:
    s = _safe_str(stem)
    if not s:
        return ""
    safe = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", ".", " "))
    safe = safe.strip().replace(" ", "_")
    safe = safe.replace(".yaml", "").replace(".yml", "")
    return safe


def _policy_editor_page(*, policy_files: list[Path], selected_policy: Path) -> None:
    """
    정책 확인/수정 페이지
    """
    from app.policy_v2 import load_policy_v2

    st.subheader("정책 미리보기 및 수정")
    st.caption("좌측에서 수정, 우측에서 미리보기. 저장 전에는 YAML 검증을 권장합니다.")

    policy_names = [p.stem for p in policy_files]
    pick = st.selectbox(
        "정책 파일 선택",
        options=policy_names,
        index=policy_names.index(selected_policy.stem) if selected_policy.stem in policy_names else 0,
        key="policy_editor_pick",
    )
    policy_path = next(p for p in policy_files if p.stem == pick)
    raw_text = policy_path.read_text(encoding="utf-8")

    st.session_state.setdefault("policy_yaml_text", raw_text)
    # policy selection changed -> refresh editor
    if st.session_state.get("_policy_editor_last_pick") != pick:
        st.session_state["_policy_editor_last_pick"] = pick
        st.session_state["policy_yaml_text"] = raw_text
        st.session_state.pop("policy_interpretation", None)
        st.session_state.pop("policy_draft_text", None)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("정책 수정")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        if c1.button("원본 다시 불러오기", use_container_width=True):
            st.session_state["policy_yaml_text"] = raw_text
            st.session_state.pop("policy_interpretation", None)
            st.rerun()

        if c2.button("YAML 검증", use_container_width=True):
            try:
                obj = yaml.safe_load(st.session_state.get("policy_yaml_text", ""))
                # validate with loader
                tmp_path = policy_path.parent / f"__tmp_validate__{policy_path.name}"
                tmp_path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
                _ = load_policy_v2(tmp_path)
                tmp_path.unlink(missing_ok=True)
                st.success("OK: YAML 파싱 및 v2 정책 로드 성공")
            except Exception as e:
                st.error(f"검증 실패: {e}")

        if c3.button("저장(덮어쓰기)", use_container_width=True, type="primary"):
            try:
                # backup
                hist = PROJECT_ROOT / "config" / "history"
                hist.mkdir(parents=True, exist_ok=True)
                backup = hist / f"{policy_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                backup.write_text(raw_text, encoding="utf-8")
                # save
                policy_path.write_text(str(st.session_state.get("policy_yaml_text", "")), encoding="utf-8")
                st.success(f"저장 완료. 백업: {backup.name}")
            except Exception as e:
                st.error(f"저장 실패: {e}")

        if c4.button("정책 해석", use_container_width=True):
            # prefer OpenAI if available; fallback to local summary
            api_key = os.getenv("OPENAI_API_KEY") or ""
            model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
            ytxt = str(st.session_state.get("policy_yaml_text", ""))
            ok, msg = _openai_healthcheck(api_key)
            if ok:
                try:
                    sys_txt = "너는 QA 디바이스 추천 정책(v2) YAML을 읽고, 사람이 이해하기 쉬운 한국어로 해석/주의점을 설명하는 도우미다."
                    usr_txt = (
                        "아래 YAML을 읽고 다음을 출력해줘:\n"
                        "1) 핵심 규칙 요약(필터/중복제거/다양성/제조사/타이브레이커)\n"
                        "2) 실제로 어떤 결과가 나올지 예시(특히 chipset 다양성/cpu_family)\n"
                        "3) 흔한 실수/주의사항\n"
                        "설명은 한국어로, 불릿 위주로.\n\n"
                        f"YAML:\n{ytxt}\n"
                    )
                    st.session_state["policy_interpretation"] = _openai_chat(
                        api_key=api_key, model=model, user_text=usr_txt, system_text=sys_txt
                    )
                except Exception as e:
                    st.session_state["policy_interpretation"] = f"(OpenAI 해석 실패) {e}\n\n[로컬 요약]\n{_policy_local_summary(ytxt)}"
            else:
                st.session_state["policy_interpretation"] = f"(OpenAI 미사용: {msg})\n\n[로컬 요약]\n{_policy_local_summary(ytxt)}"

        st.text_area("정책 YAML", key="policy_yaml_text", height=420)

        if st.session_state.get("policy_interpretation"):
            st.markdown("#### 정책 해석")
            st.write(st.session_state.get("policy_interpretation"))

    st.divider()
    st.subheader("OpenAI로 정책 만들기/해석 도움")
    st.caption("OpenAI 키는 사이드바의 'API 연결 설정(세션/저장)'에서 적용하세요.")

    # prompt templates
    with st.expander("프롬프트 템플릿(복사해서 사용)", expanded=False):
        st.code(
            "\n".join(
                [
                    "목표: QA 디바이스 추천 v2 정책 YAML을 만든다.",
                    "",
                    "요구사항을 아래 항목으로 써줘:",
                    "- 프로젝트: KP/KRJP/PALM",
                    "- 플랫폼: android",
                    "- Rank별 목표 수량 예시",
                    "- 중복제거(기본): (cpu,gpu,ram_gb) 프로파일 대표",
                    "- 추가 커버리지: diversity.must_cover (예: cpu_family, gpu_family, display_bucket 등)",
                    "- 제조사 정책: manufacturer_policy.within_rank.mode=soft_dedupe, penalty_weight=0.5",
                    "- tie_breakers: newest_release_year 등",
                    "",
                    "출력: v2 정책 YAML만(설명 금지).",
                ]
            ),
            language="text",
        )

    natural = st.text_area(
        "정책 요구(자연어) / 질문",
        value="",
        placeholder="예) A+에서 2대면 Snapdragon 1대 + Exynos 1대는 꼭 포함. 제조사 중복은 완화. display_bucket은 optional.",
        height=120,
    )
    colg1, colg2 = st.columns(2)
    if colg1.button("YAML 초안 생성", use_container_width=True):
        try:
            api_key = os.getenv("OPENAI_API_KEY") or ""
            model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
            sys_txt = "너는 QA 디바이스 추천 정책(v2)을 YAML로 작성하는 도우미다. 반드시 YAML만 출력하고, 불필요한 설명은 하지 마라."
            usr_txt = (
                "아래 요구사항을 v2 정책 YAML로 변환해줘.\n"
                "규칙:\n"
                "- YAML 최상위는 project/version/candidate_filter/dedupe/diversity/tie_breakers/manufacturer_policy/versioning/focus 등으로 구성\n"
                "- diversity.within_rank.must_cover를 적극 사용해서 chipset 다양성(cpu_family)을 반영\n"
                "- 가능한 한 기존 KP/KRJP/PALM 스타일을 유지\n\n"
                f"요구사항:\n{natural}\n"
            )
            out = _openai_chat(api_key=api_key, model=model, user_text=usr_txt, system_text=sys_txt)
            # best-effort: validate yaml
            obj = yaml.safe_load(out)
            draft = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
            st.session_state["policy_draft_text"] = draft
            st.session_state["policy_yaml_text"] = draft
            st.success("초안을 에디터에 반영했습니다. 아래에서 새 파일명으로 저장할 수 있습니다.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if colg2.button("현재 정책 해석 도움", use_container_width=True):
        try:
            api_key = os.getenv("OPENAI_API_KEY") or ""
            model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
            sys_txt = (
                "너는 QA 디바이스 추천 정책(v2) YAML을 읽고, "
                "정책이 '실제로 사용하는 조건/키'를 짧고 정확하게 정리하는 도우미다. "
                "장문 설명/예시/주의사항/추측은 금지한다."
            )
            usr_txt = (
                "아래 YAML에서 '조건으로 실제 사용되는 항목'만 추출해서, 한국어로 매우 짧게 요약해줘.\n"
                "출력 형식(반드시 준수):\n"
                "- 각 줄: `YAML경로: 현재값 — 한줄 의미`\n"
                "- 섹션은 아래 순서로만 정리:\n"
                "  candidate_filter / dedupe / diversity / manufacturer_policy / tie_breakers / versioning / focus\n"
                "- 최대 15줄\n"
                "- 예시/주의사항/긴 설명/시뮬레이션 금지\n\n"
                f"YAML:\n{st.session_state.get('policy_yaml_text','')}\n"
            )
            out = _openai_chat(api_key=api_key, model=model, user_text=usr_txt, system_text=sys_txt)
            st.write(out)
        except Exception as e:
            st.error(str(e))

    # draft save-as flow
    if st.session_state.get("policy_draft_text"):
        st.markdown("#### YAML 초안 저장(새 파일)")
        default_name = f"{policy_path.stem}_draft_{datetime.now().strftime('%m%d_%H%M')}"
        new_name = st.text_input("새 정책 파일명(.yaml 없이)", value=default_name, key="new_policy_name")
        new_stem = _safe_policy_filename(new_name)
        csa1, csa2 = st.columns([1, 2])
        if csa1.button("새 파일로 저장", use_container_width=True):
            if not new_stem:
                st.error("파일명이 비어 있습니다.")
            else:
                out_path = CONFIG_V2_DIR / f"{new_stem}.yaml"
                if out_path.exists():
                    st.error(f"이미 존재하는 파일명입니다: {out_path.name}")
                else:
                    try:
                        # validate yaml
                        _ = yaml.safe_load(st.session_state.get("policy_yaml_text", ""))
                        out_path.write_text(str(st.session_state.get("policy_yaml_text", "")), encoding="utf-8")
                        st.success(f"저장 완료: {out_path.name} (정책 선택 목록에 곧 나타납니다)")
                    except Exception as e:
                        st.error(f"저장 실패: {e}")
        with csa2:
            st.caption("저장 후 상단의 '정책 파일 선택'에서 새 파일을 선택할 수 있습니다.")

    with right:
        st.subheader("정책 미리보기")
        st.code(str(st.session_state.get("policy_yaml_text", "")), language="yaml")
        try:
            obj = yaml.safe_load(str(st.session_state.get("policy_yaml_text", "")))
            if isinstance(obj, dict):
                st.markdown("#### 요약(자동)")
                st.write(
                    {
                        "project": obj.get("project"),
                        "candidate_filter": obj.get("candidate_filter"),
                        "dedupe": obj.get("dedupe"),
                        "diversity": obj.get("diversity"),
                        "manufacturer_policy": obj.get("manufacturer_policy"),
                        "tie_breakers": obj.get("tie_breakers"),
                    }
                )
        except Exception:
            pass


def _run() -> None:
    st.set_page_config(page_title="devices_auto - Policy v2 POC", layout="wide")
    st.title("Policy-first 디바이스 자동 추출 (POC)")
    st.caption("v2 정책(KRJP/KP/PALM) + Rank 수량 입력 → 중복 제거 → 부족/로그 → 멀티시트 엑셀")
    # (요청) build/python 표시 제거

    _ensure_repo_root_on_path()

    from app.policy_v2 import load_policy_v2
    from app.policy_v2_engine import PolicyV2RunInputs, run_policy_v2
    from app.testbed_normalizer import load_testbed, normalize_testbed
    from app.env import load_default_env

    # load local.env/.env best-effort so keys are available
    load_default_env(PROJECT_ROOT, override=False)

    # --- API 연결 설정(세션/저장) ---
    st.sidebar.header("API 연결 설정(세션/저장)")
    local_env_path = PROJECT_ROOT / "local.env"
    existing = _read_env_kv(local_env_path)

    with st.sidebar.expander("OpenAI / Jira 키 설정", expanded=False):
        st.caption("세션 적용: 현재 실행 중인 Streamlit 프로세스에만 반영됩니다. 저장을 켜면 local.env에 기록됩니다.")

        openai_key = st.text_input(
            "OPENAI_API_KEY",
            value=os.getenv("OPENAI_API_KEY") or existing.get("OPENAI_API_KEY", ""),
            type="password",
        )
        openai_model = st.text_input(
            "OPENAI_MODEL(선택)",
            value=os.getenv("OPENAI_MODEL") or existing.get("OPENAI_MODEL", "gpt-4.1-mini"),
        )

        st.divider()
        jira_base = st.text_input(
            "JIRA_BASE_URL(선택)",
            value=os.getenv("JIRA_BASE_URL") or existing.get("JIRA_BASE_URL", ""),
            placeholder="https://xxx.atlassian.net",
        )
        jira_email = st.text_input(
            "JIRA_EMAIL(선택)",
            value=os.getenv("JIRA_EMAIL") or existing.get("JIRA_EMAIL", ""),
        )
        jira_token = st.text_input(
            "JIRA_API_TOKEN(선택)",
            value=os.getenv("JIRA_API_TOKEN") or existing.get("JIRA_API_TOKEN", ""),
            type="password",
        )

        # status badge
        st.session_state.setdefault("api_status", "미적용")
        st.info(f"상태: {st.session_state.get('api_status')}")

        save_local = st.checkbox("local.env에 저장", value=True)
        c_ap1, c_ap2 = st.columns(2)
        if c_ap1.button("세션 적용", use_container_width=True):
            if _safe_str(openai_key):
                os.environ["OPENAI_API_KEY"] = openai_key.strip()
            if _safe_str(openai_model):
                os.environ["OPENAI_MODEL"] = openai_model.strip()
            if _safe_str(jira_base):
                os.environ["JIRA_BASE_URL"] = jira_base.strip()
            if _safe_str(jira_email):
                os.environ["JIRA_EMAIL"] = jira_email.strip()
            if _safe_str(jira_token):
                os.environ["JIRA_API_TOKEN"] = jira_token.strip()

            if save_local:
                _write_env_kv(
                    local_env_path,
                    {
                        "OPENAI_API_KEY": openai_key.strip(),
                        "OPENAI_MODEL": openai_model.strip(),
                        "JIRA_BASE_URL": jira_base.strip(),
                        "JIRA_EMAIL": jira_email.strip(),
                        "JIRA_API_TOKEN": jira_token.strip(),
                    },
                )
                # healthcheck
                ok, msg = _openai_healthcheck(openai_key)
                st.session_state["api_status"] = "정상" if ok else f"비정상({msg})"
                st.success("세션 적용 + local.env 저장 완료")
            else:
                ok, msg = _openai_healthcheck(openai_key)
                st.session_state["api_status"] = "정상" if ok else f"비정상({msg})"
                st.success("세션 적용 완료(저장은 안 함)")
            st.rerun()
        if c_ap2.button("local.env 다시 읽기", use_container_width=True):
            st.rerun()

    st.sidebar.divider()

    st.sidebar.header("입력")
    policy_files = sorted(CONFIG_V2_DIR.glob("*.yaml"))
    if not policy_files:
        st.sidebar.error("config/policies_v2/*.yaml 이 없습니다.")
        return
    policy_names = [p.stem for p in policy_files]
    # lightweight session defaults (no file presets)
    st.session_state.setdefault("policy_pick", policy_names[0] if policy_names else "")
    st.session_state.setdefault("version", "4.3.0")
    st.session_state.setdefault("rk_Aplus", 0)
    st.session_state.setdefault("rk_A", 0)
    st.session_state.setdefault("rk_B", 0)
    st.session_state.setdefault("rk_C", 0)
    st.session_state.setdefault("rk_D", 0)
    st.session_state.setdefault("rk_dash", 0)
    st.session_state.setdefault("required_nos", "")
    st.session_state.setdefault("prev_excl_nos", "")

    # page mode
    st.session_state.setdefault("page_mode", "run")

    policy_pick = st.sidebar.selectbox(
        "프로젝트 정책(v2)",
        options=policy_names,
        index=policy_names.index(st.session_state["policy_pick"]) if st.session_state["policy_pick"] in policy_names else 0,
        key="policy_pick",
    )
    policy_path = next(p for p in policy_files if p.stem == policy_pick)
    policy = load_policy_v2(policy_path)

    # bottom button will toggle this, but also allow in sidebar
    if st.sidebar.button("정책확인(미리보기/수정)", use_container_width=True):
        st.session_state["page_mode"] = "policy"
        st.rerun()

    if st.session_state.get("page_mode") == "policy":
        if st.button("← 추천 실행으로 돌아가기"):
            st.session_state["page_mode"] = "run"
            st.rerun()
        _policy_editor_page(policy_files=policy_files, selected_policy=policy_path)
        return

    version = st.sidebar.text_input("릴리즈 버전", value=st.session_state.get("version", "4.3.0"), key="version")

    # upload first so we can build dropdown options (target country, etc.)
    st.sidebar.divider()
    upload = st.sidebar.file_uploader("마스터 디바이스 목록 업로드(Excel/CSV)", type=["xlsx", "xls", "csv"])

    platforms = _parse_platforms(getattr(policy.candidate_filter, "platform", "android"))

    def _rank_inputs(prefix: str) -> dict[str, int]:
        # (요청) 문구 정리: 플랫폼별 합계는 아래 caption으로 표시
        st.sidebar.subheader("Rank별 목표 수량")
        # (요청) 표시 순서: A+, A, B, C, D, 무등급 (단, iOS 탭은 D 제거)
        top = st.sidebar.columns(3)
        with top[0]:
            a_plus = int(st.number_input("A+", min_value=0, step=1, value=int(st.session_state.get(f"rk_{prefix}_Aplus", 0) or 0), key=f"rk_{prefix}_Aplus"))
        with top[1]:
            a = int(st.number_input("A", min_value=0, step=1, value=int(st.session_state.get(f"rk_{prefix}_A", 0) or 0), key=f"rk_{prefix}_A"))
        with top[2]:
            b = int(st.number_input("B", min_value=0, step=1, value=int(st.session_state.get(f"rk_{prefix}_B", 0) or 0), key=f"rk_{prefix}_B"))

        bot = st.sidebar.columns(3)
        with bot[0]:
            c = int(st.number_input("C", min_value=0, step=1, value=int(st.session_state.get(f"rk_{prefix}_C", 0) or 0), key=f"rk_{prefix}_C"))
        with bot[1]:
            if prefix.lower() == "ios":
                # (요청) iOS 탭에서는 D 등급 제거(항상 0)
                d = 0
                st.caption("D: (iOS 입력 없음)")
            else:
                d = int(st.number_input("D", min_value=0, step=1, value=int(st.session_state.get(f"rk_{prefix}_D", 0) or 0), key=f"rk_{prefix}_D"))
        with bot[2]:
            dash = int(st.number_input("무등급", min_value=0, step=1, value=int(st.session_state.get(f"rk_{prefix}_dash", 0) or 0), key=f"rk_{prefix}_dash"))
        rt = {"A+": a_plus, "A": a, "B": b, "C": c, "D": d, "-": dash}
        tn = int(sum(rt.values()))
        # (요청) Top N/순서 문구 삭제, 합계만 플랫폼별로 표시
        platform_label = "AOS" if prefix.lower() == "android" else "IOS"
        st.sidebar.caption(f"{platform_label} Rank별 목표 수량(합계={tn})")
        return rt

    # single-platform default keys 유지 (기존 세션 값 호환)
    if len(platforms) == 1:
        prefix = "android" if platforms[0] == "android" else "ios"
        # migrate old keys once (best-effort)
        if prefix == "android":
            for old, new in [
                ("rk_Aplus", "rk_android_Aplus"),
                ("rk_A", "rk_android_A"),
                ("rk_B", "rk_android_B"),
                ("rk_C", "rk_android_C"),
                ("rk_D", "rk_android_D"),
                ("rk_dash", "rk_android_dash"),
            ]:
                if old in st.session_state and new not in st.session_state:
                    st.session_state[new] = st.session_state.get(old)
        rank_targets_android = _rank_inputs(prefix)
        rank_targets_ios = {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0, "-": 0}
    else:
        st.sidebar.subheader("Rank별 목표 수량 - 플랫폼별")
        tab_a, tab_i = st.sidebar.tabs(["Android(AOS)", "iOS"])
        with tab_a:
            rank_targets_android = _rank_inputs("android")
        with tab_i:
            rank_targets_ios = _rank_inputs("ios")

    # target exclusion (optional)
    # (요청) 볼드체(서브헤더) 제거
    st.sidebar.caption("타겟 국가 제외(선택)")
    excl_default: list[str] = []
    opts: list[str] = []
    if upload is not None:
        try:
            opts = _extract_target_country_options_from_upload(getattr(upload, "name", ""), bytes(upload.getbuffer()))
        except Exception:
            opts = []
    if not opts:
        st.sidebar.caption("업로드 파일에서 '타겟 국가' 컬럼을 찾지 못했습니다. (선택하지 않으면 제외 없음)")
    exclude_targets = st.sidebar.multiselect(
        "제외할 타겟 국가(중복 선택)",
        options=opts,
        default=[x for x in excl_default if x in opts],
    )

    # note exclusion (optional)
    # (요청) 볼드체(서브헤더) 제거
    st.sidebar.caption("NOTE 제외(선택)")
    note_opts: list[str] = []
    if upload is not None:
        try:
            note_opts = _extract_note_options_from_upload(getattr(upload, "name", ""), bytes(upload.getbuffer()))
        except Exception:
            note_opts = []
    if not note_opts:
        st.sidebar.caption("업로드 파일에서 'NOTE/비고' 컬럼을 찾지 못했습니다. (선택하지 않으면 제외 없음)")
    exclude_notes = st.sidebar.multiselect(
        "제외할 NOTE(중복 선택)",
        options=note_opts,
        default=[],
    )

    required_nos = st.sidebar.text_input("필수 디바이스 No(콤마, 선택)", value=st.session_state.get("required_nos", ""), key="required_nos")
    prev_excl_nos = st.sidebar.text_input("직전 버전 제외 No(콤마, 선택)", value=st.session_state.get("prev_excl_nos", ""), key="prev_excl_nos", help="해당 No들의 제품명/모델명은 이번 버전에서 제외")

    st.sidebar.divider()
    run_btn = st.sidebar.button("자동 추출 실행", type="primary", disabled=(upload is None))

    st.subheader("결과")
    result_box = st.empty()
    download_box = st.empty()

    if not run_btn:
        return

    total_n_android = int(sum(rank_targets_android.values())) if isinstance(rank_targets_android, dict) else 0
    total_n_ios = int(sum(rank_targets_ios.values())) if isinstance(rank_targets_ios, dict) else 0
    total_n_all = total_n_android + total_n_ios if len(platforms) == 2 else (total_n_android if platforms[0] == "android" else total_n_ios)
    if total_n_all <= 0:
        st.sidebar.error("Rank 목표 수량 합계가 0입니다.")
        return

    with st.spinner("업로드 파일 로딩/정규화 중..."):
        up_path = _save_upload(upload, prefix=f"testbed_{policy.project}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        raw_df = load_testbed(up_path)
        norm_df = normalize_testbed(raw_df)

    inputs = PolicyV2RunInputs(
        project=policy.project,
        version=version,
        rank_targets=rank_targets_android if (len(platforms) == 1 and platforms[0] == "android") else rank_targets_android,
        required_nos=_split_csv_list(required_nos),
        exclude_prev_version_nos=_split_csv_list(prev_excl_nos),
        exclude_target_countries=list(exclude_targets or []),
        exclude_notes=list(exclude_notes or []),
    )

    with st.spinner("정책 기반 자동 추출 중..."):
        # 플랫폼이 android,ios 모두면 각각 따로 추출
        if len(platforms) == 2:
            from dataclasses import replace
            from app.policy_v2 import CandidateFilter

            cf0 = policy.candidate_filter

            p_android = replace(policy, candidate_filter=CandidateFilter(platform="android", required_fields=cf0.required_fields, availability=cf0.availability))
            p_ios = replace(policy, candidate_filter=CandidateFilter(platform="ios", required_fields=cf0.required_fields, availability=cf0.availability))

            out_android = run_policy_v2(master_df=norm_df, policy=p_android, inputs=replace(inputs, rank_targets=rank_targets_android))
            out_ios = run_policy_v2(master_df=norm_df, policy=p_ios, inputs=replace(inputs, rank_targets=rank_targets_ios))
        else:
            out_android = run_policy_v2(master_df=norm_df, policy=policy, inputs=inputs)
            out_ios = None

    with result_box.container():
        if out_ios is None:
            st.subheader("Rank별 최종 선택(미리보기)")
            sel_with_why = _attach_why(out_android.selected, out_android.decision_log)
            cleaned = _dedupe_cols(_clean_result_df(sel_with_why))
            # debug: if No looks empty, show what columns exist
            try:
                no_series = cleaned["No"].fillna("").astype(str).str.strip() if "No" in cleaned.columns else pd.Series([""] * len(cleaned), index=cleaned.index)
                no_empty = ("No" not in cleaned.columns) or no_series.eq("").all()
                no_tmp = no_series.str.startswith("TMP")
            except Exception:
                no_empty = True
                no_tmp = False
            if no_empty or no_tmp:
                with st.expander("디버그: No 컬럼이 비어있는 이유 확인", expanded=False):
                    st.write(
                        {
                            "selected_columns": list(sel_with_why.columns)[:80],
                            "cleaned_columns": list(cleaned.columns)[:80],
                            "no_candidates_norm_keys": [
                                str(c) for c in sel_with_why.columns if "no" in str(c).lower() or "번호" in str(c)
                            ][:40],
                            "no_preview": no_series.head(10).tolist() if "No" in cleaned.columns else [],
                        }
                    )
                    # show sample values for likely columns + pattern match counts
                    no_re = re.compile(r"(?i)^(AP|IP|AT|IT|R)\\d+")
                    likely = [c for c in sel_with_why.columns if "no" in str(c).lower() or "번호" in str(c) or "관리" in str(c)]
                    for c in likely[:12]:
                        try:
                            s = sel_with_why[c].fillna("").astype(str).str.strip()
                            non_empty = int(s.ne("").sum())
                            pat = int(s.map(lambda x: bool(no_re.match(str(x)))).sum())
                            st.write({f"col={c} non_empty": non_empty, "pattern(AP/IP/AT/IT/R\\d+)": pat, "sample": s[s.ne("")].head(5).tolist()})
                        except Exception:
                            pass
            st.dataframe(cleaned.head(50), use_container_width=True)
            st.subheader("부족 요약")
            ss = out_android.shortage_summary if out_android.shortage_summary is not None else pd.DataFrame()
            if ss.empty:
                st.info("부족 없음(모든 Rank 목표 수량을 충족했습니다).")
            else:
                st.dataframe(_dedupe_cols(ss), use_container_width=True)
        else:
            t1, t2 = st.tabs(["Android(AOS) 결과", "iOS 결과"])
            with t1:
                st.subheader("Rank별 최종 선택(미리보기)")
                sel_with_why = _attach_why(out_android.selected, out_android.decision_log)
                cleaned = _dedupe_cols(_clean_result_df(sel_with_why))
                st.dataframe(cleaned.head(50), use_container_width=True)
                st.subheader("부족 요약")
                ss = out_android.shortage_summary if out_android.shortage_summary is not None else pd.DataFrame()
                if ss.empty:
                    st.info("부족 없음")
                else:
                    st.dataframe(_dedupe_cols(ss), use_container_width=True)
            with t2:
                st.subheader("Rank별 최종 선택(미리보기)")
                sel_with_why = _attach_why(out_ios.selected, out_ios.decision_log)
                cleaned = _dedupe_cols(_clean_result_df(sel_with_why))
                st.dataframe(cleaned.head(50), use_container_width=True)
                st.subheader("부족 요약")
                ss = out_ios.shortage_summary if out_ios.shortage_summary is not None else pd.DataFrame()
                if ss.empty:
                    st.info("부족 없음")
                else:
                    st.dataframe(_dedupe_cols(ss), use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if out_ios is None:
            _dedupe_cols(out_android.rule_summary).to_excel(writer, sheet_name="Policy_Summary", index=False)
            _dedupe_cols(out_android.shortage_summary).to_excel(writer, sheet_name="Shortage", index=False)
            for rk in ["A+", "A", "B", "C", "D", "-"]:
                df_rk = out_android.selected_by_rank.get(rk, pd.DataFrame())
                name = f"Selected_{rk}".replace("+", "PLUS").replace("-", "DASH")
                df_rk2 = _attach_why(df_rk, out_android.decision_log)
                _dedupe_cols(_clean_result_df(df_rk2)).to_excel(writer, sheet_name=name[:31], index=False)
            all2 = _attach_why(out_android.selected, out_android.decision_log)
            _dedupe_cols(_clean_result_df(all2)).to_excel(writer, sheet_name="Selected_All", index=False)
        else:
            _dedupe_cols(out_android.rule_summary).to_excel(writer, sheet_name="Policy_Summary", index=False)
            # (요청) Shortage_AOS / Shortage_IOS 시트 생성 제거
            # (요청) AOS_DASH(-), IOS_D, IOS_DASH(-) 시트 생성 제거
            for rk in ["A+", "A", "B", "C", "D"]:
                df_rk = out_android.selected_by_rank.get(rk, pd.DataFrame())
                name = f"AOS_{rk}".replace("+", "PLUS").replace("-", "DASH")
                df_rk2 = _attach_why(df_rk, out_android.decision_log)
                _dedupe_cols(_clean_result_df(df_rk2)).to_excel(writer, sheet_name=name[:31], index=False)
            for rk in ["A+", "A", "B", "C"]:
                df_rk = out_ios.selected_by_rank.get(rk, pd.DataFrame())
                name = f"IOS_{rk}".replace("+", "PLUS").replace("-", "DASH")
                df_rk2 = _attach_why(df_rk, out_ios.decision_log)
                _dedupe_cols(_clean_result_df(df_rk2)).to_excel(writer, sheet_name=name[:31], index=False)
            all_a = _attach_why(out_android.selected, out_android.decision_log)
            all_i = _attach_why(out_ios.selected, out_ios.decision_log)
            _dedupe_cols(_clean_result_df(all_a)).to_excel(writer, sheet_name="AOS_Selected_All", index=False)
            _dedupe_cols(_clean_result_df(all_i)).to_excel(writer, sheet_name="IOS_Selected_All", index=False)

    with download_box.container():
        st.subheader("다운로드")
        st.download_button(
            "엑셀 다운로드(멀티시트)",
            data=buf.getvalue(),
            file_name=f"{policy.project}_{version}_selection.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        # (요청) 로그 다운로드 버튼 제거(엑셀 다운로드 1개만)

    # bottom navigation
    st.divider()
    # (요청) 결과 화면의 정책확인 버튼 제거


if __name__ == "__main__":
    _run()


