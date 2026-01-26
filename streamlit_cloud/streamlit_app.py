from __future__ import annotations

import io
import os
import re
import sys
import time
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


def _unique_preserve_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        k = str(x).strip()
        if not k:
            continue
        kk = k.upper()
        if kk in seen:
            continue
        seen.add(kk)
        out.append(k)
    return out


def _example_prompts_10() -> list[str]:
    """
    '정책 프롬프트 확인/입력' placeholder에 맞춘 예시 10개.
    - 검증/디버그(버튼 1번)용으로도 사용한다.
    """
    return [
        "Android만 + 삼성만 + GPU에 PowerVR 포함 + RAM 3~8GB",
        "iOS만 + OS 15 이상 + GPU에 Apple 포함",
        "Android만 + 제조사 LG,SAMSUNG + OS 10 이상 + GPU에 Adreno 포함",
        "Android,iOS + RAM 2~3GB만 보되, 예외 OR(any_of)로 GPU에 PowerVR 포함 기기도 함께 포함",
        "64bit만 + OS 9 이상 + GPU에 Mali 포함",
        "32bit만 + RAM 1~2GB만",
        "제조사 GOOGLE만 + Android만 + OS 12 이상",
        "제조사 APPLE만 + iOS만 + RAM 4~12GB",
        "GPU에 Adreno 포함만 보되, 예외 OR(any_of)로 GPU에 Mali 포함 기기도 함께 포함",
        "제조사 SAMSUNG,APPLE + RAM 3~6GB + OS 10 이상 + (예외 OR(any_of)) GPU에 PowerVR 포함도 추가 포함",
    ]


def _auto_textarea_height(text: str, *, min_h: int = 160, max_h: int = 520) -> int:
    """
    Streamlit text_area는 자동 높이 조절이 없어, 내용 줄 수 기준으로 height를 계산한다.
    (완전한 word-wrap 추정은 어렵기 때문에 줄바꿈 기준 + 최소/최대 clamp)
    """
    s = str(text or "")
    lines = max(1, s.count("\n") + 1)
    # 1줄당 약 22px + padding
    h = 80 + (lines * 22)
    return int(max(min_h, min(max_h, h)))


def _prefill_policy_edit_ui_from_yaml(*, obj: dict[str, Any], policy_stem: str) -> None:
    """
    정책 수정(UI) 화면에서 새 정책 만들기와 동일한 폼 구성을 사용하기 위해,
    기존 YAML에서 값을 읽어 session_state(UI 키)에 채워 넣는다.
    (지원 범위: platform, manufacturer, architecture, gpu contains, ram_gb range, os_ver min, any_of(gpu 예외 1개))
    """
    # Normalize legacy/all_of formats so UI reflects what the engine actually supports.
    try:
        obj_norm, _warns = _normalize_policy_obj_to_supported_v2(dict(obj or {}))
        if isinstance(obj_norm, dict):
            obj = obj_norm
    except Exception:
        pass

    cf = obj.get("candidate_filter") if isinstance(obj.get("candidate_filter"), dict) else {}
    platform_txt = str((cf or {}).get("platform") or "")
    plats = _parse_platforms(platform_txt) if platform_txt else []
    if not plats:
        plats = ["android", "ios"]
    st.session_state["ui_cf_platforms_edit"] = plats

    def _as_str_list(v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        return [s] if s else []

    include_values = (cf or {}).get("include_values") if isinstance((cf or {}).get("include_values"), dict) else {}
    include_contains = (cf or {}).get("include_contains") if isinstance((cf or {}).get("include_contains"), dict) else {}
    numeric_ranges = (cf or {}).get("numeric_ranges") if isinstance((cf or {}).get("numeric_ranges"), dict) else {}
    min_values = (cf or {}).get("min_values") if isinstance((cf or {}).get("min_values"), dict) else {}

    # If A∪B(any_of) 형태라면, base 조건은 any_of[0]에 들어가 있을 수 있다.
    any_of = (cf or {}).get("any_of")
    base_br: dict[str, Any] | None = None
    gpu_br: dict[str, Any] | None = None
    if isinstance(any_of, (list, tuple)) and any_of:
        # base branch: prefer first dict branch
        for br in any_of:
            if isinstance(br, dict):
                base_br = br
                break

        # gpu exception branch: prefer "gpu only" branch
        def _is_gpu_only_branch(br: dict[str, Any]) -> bool:
            icb = br.get("include_contains") if isinstance(br.get("include_contains"), dict) else {}
            if not isinstance(icb, dict) or not _as_str_list(icb.get("gpu")):
                return False
            ic2 = dict(icb)
            ic2.pop("gpu", None)
            has_other_ic = bool(ic2)
            has_iv = isinstance(br.get("include_values"), dict) and bool(br.get("include_values"))
            has_nr = isinstance(br.get("numeric_ranges"), dict) and bool(br.get("numeric_ranges"))
            has_mv = isinstance(br.get("min_values"), dict) and bool(br.get("min_values"))
            return (not has_other_ic) and (not has_iv) and (not has_nr) and (not has_mv)

        for br in any_of:
            if isinstance(br, dict) and _is_gpu_only_branch(br):
                gpu_br = br
                break
        if gpu_br is None:
            for br in any_of:
                if not isinstance(br, dict):
                    continue
                icb = br.get("include_contains") if isinstance(br.get("include_contains"), dict) else {}
                if isinstance(icb, dict) and _as_str_list(icb.get("gpu")):
                    gpu_br = br
                    break

    # When base branch exists, read base-filter fields from it (architecture/manufacturer/ram/os/gpu_basic)
    if isinstance(base_br, dict):
        include_values = base_br.get("include_values") if isinstance(base_br.get("include_values"), dict) else include_values
        include_contains = base_br.get("include_contains") if isinstance(base_br.get("include_contains"), dict) else include_contains
        numeric_ranges = base_br.get("numeric_ranges") if isinstance(base_br.get("numeric_ranges"), dict) else numeric_ranges
        min_values = base_br.get("min_values") if isinstance(base_br.get("min_values"), dict) else min_values

    manu = _as_str_list((include_values or {}).get("manufacturer"))
    st.session_state["ui_cf_manufacturer_edit"] = ", ".join(manu) if manu else ""

    arch = _as_str_list((include_values or {}).get("architecture"))
    # 둘 다 체크면 "제한 없음"과 동일하게 취급
    arch_norm = [a.strip().lower() for a in (arch or []) if str(a).strip()]
    st.session_state["ui_cf_arch_64_edit"] = ("64bit" in arch_norm) or (not arch_norm)
    st.session_state["ui_cf_arch_32_edit"] = ("32bit" in arch_norm) or (not arch_norm)

    gpu_basic = _as_str_list((include_contains or {}).get("gpu"))
    st.session_state["ui_cf_gpu_contains_edit"] = gpu_basic[0] if gpu_basic else ""

    # RAM range
    rr = (numeric_ranges or {}).get("ram_gb")
    if isinstance(rr, (list, tuple)) and len(rr) >= 2:
        try:
            lo = int(float(rr[0]))
            hi = int(float(rr[1]))
            st.session_state["ui_cf_ram_rng_on_edit"] = True
            st.session_state["ui_cf_ram_lo_edit"] = lo
            st.session_state["ui_cf_ram_hi_edit"] = hi
        except Exception:
            st.session_state["ui_cf_ram_rng_on_edit"] = False
    else:
        st.session_state["ui_cf_ram_rng_on_edit"] = False

    # OS min
    os_v = (min_values or {}).get("os_ver", None)
    if os_v is not None and str(os_v).strip():
        st.session_state["ui_cf_os_min_on_edit"] = True
        st.session_state["ui_cf_os_min_text_edit"] = str(os_v).strip().rstrip(".0")
    else:
        st.session_state["ui_cf_os_min_on_edit"] = False
        st.session_state["ui_cf_os_min_text_edit"] = ""

    # any_of (GPU 예외 1개만 지원) — A∪B에서는 GPU 예외가 any_of[1]일 수 있다.
    any_gpu = ""
    if isinstance(gpu_br, dict):
        ic0 = gpu_br.get("include_contains") if isinstance(gpu_br.get("include_contains"), dict) else {}
        g0 = _as_str_list((ic0 or {}).get("gpu"))
        if g0:
            any_gpu = g0[0]
    st.session_state["ui_cf_gpu_or_on_edit"] = bool(any_gpu)
    st.session_state["ui_cf_any2_gpu_edit"] = any_gpu or ""

    # meta fields
    # - "프로젝트 정책 이름"은 정책 파일명(.yaml 없이)으로 통일한다.
    st.session_state["ui_policy_name_edit"] = str(policy_stem).strip()
    st.session_state["ui_policy_desc_edit"] = str(obj.get("description") or "").strip()


def _apply_policy_edit_ui_to_obj(*, base_obj: dict[str, Any], ui_project: str) -> dict[str, Any]:
    """
    edit UI 폼 값들을 base_obj에 병합/반영해서 최종 저장용 obj(dict)를 만든다.
    - project/version/description은 없으면 주입
    - candidate_filter의 일부 필드(platform/manufacturer/architecture/gpu/ram/os/any_of)는 UI를 기준으로 반영(없으면 제거)
    """
    obj: dict[str, Any] = dict(base_obj or {})
    obj["project"] = str(obj.get("project") or ui_project or "NEWPROJ").strip()
    obj["version"] = str(obj.get("version") or "v2").strip() or "v2"
    ui_desc = str(st.session_state.get("ui_policy_desc_edit") or "").strip()
    if ui_desc:
        obj["description"] = ui_desc

    cf = obj.get("candidate_filter")
    if not isinstance(cf, dict):
        cf = {}
        obj["candidate_filter"] = cf

    # platform
    plats = st.session_state.get("ui_cf_platforms_edit") or []
    if isinstance(plats, (list, tuple)) and plats:
        cf["platform"] = ",".join([str(x).strip() for x in plats if str(x).strip()])

    # include_values
    iv = cf.get("include_values")
    if not isinstance(iv, dict):
        iv = {}
        cf["include_values"] = iv

    manu_txt = str(st.session_state.get("ui_cf_manufacturer_edit") or "").strip()
    manu = _unique_preserve_order(_split_csv_list(manu_txt))
    if manu:
        iv["manufacturer"] = [x.upper() for x in manu]
    else:
        iv.pop("manufacturer", None)

    arch_sel: list[str] = []
    if bool(st.session_state.get("ui_cf_arch_64_edit")):
        arch_sel.append("64bit")
    if bool(st.session_state.get("ui_cf_arch_32_edit")):
        arch_sel.append("32bit")
    if len(arch_sel) == 1:
        iv["architecture"] = [arch_sel[0]]
    else:
        iv.pop("architecture", None)
    if not iv:
        cf.pop("include_values", None)

    # include_contains
    ic = cf.get("include_contains")
    if not isinstance(ic, dict):
        ic = {}
        cf["include_contains"] = ic
    gpu_basic = str(st.session_state.get("ui_cf_gpu_contains_edit") or "").strip()
    if gpu_basic:
        ic["gpu"] = [gpu_basic]
    else:
        ic.pop("gpu", None)
    if not ic:
        cf.pop("include_contains", None)

    # numeric_ranges
    nr = cf.get("numeric_ranges")
    if not isinstance(nr, dict):
        nr = {}
        cf["numeric_ranges"] = nr
    if bool(st.session_state.get("ui_cf_ram_rng_on_edit")):
        try:
            lo = float(st.session_state.get("ui_cf_ram_lo_edit", 2) or 2)
            hi = float(st.session_state.get("ui_cf_ram_hi_edit", 3) or 3)
            if lo > hi:
                lo, hi = hi, lo
            nr["ram_gb"] = [lo, hi]
        except Exception:
            nr.pop("ram_gb", None)
    else:
        nr.pop("ram_gb", None)
    if not nr:
        cf.pop("numeric_ranges", None)

    # min_values
    mv = cf.get("min_values")
    if not isinstance(mv, dict):
        mv = {}
        cf["min_values"] = mv
    if bool(st.session_state.get("ui_cf_os_min_on_edit")):
        os_txt = str(st.session_state.get("ui_cf_os_min_text_edit") or "").strip()
        try:
            mv["os_ver"] = float(os_txt)
        except Exception:
            mv.pop("os_ver", None)
    else:
        mv.pop("os_ver", None)
    if not mv:
        cf.pop("min_values", None)

    # any_of (예외 GPU 1개) — 예외도 기본 조건을 반드시 만족
    if bool(st.session_state.get("ui_cf_gpu_or_on_edit")):
        any2 = str(st.session_state.get("ui_cf_any2_gpu_edit") or "").strip()
        if any2:
            # base branch: current AND filters (including basic GPU filter)
            base_branch: dict[str, Any] = {}
            if isinstance(cf.get("include_values"), dict) and cf.get("include_values"):
                base_branch["include_values"] = dict(cf.get("include_values") or {})
            if isinstance(cf.get("include_contains"), dict) and cf.get("include_contains"):
                base_branch["include_contains"] = dict(cf.get("include_contains") or {})
            if isinstance(cf.get("numeric_ranges"), dict) and cf.get("numeric_ranges"):
                base_branch["numeric_ranges"] = dict(cf.get("numeric_ranges") or {})
            if isinstance(cf.get("min_values"), dict) and cf.get("min_values"):
                base_branch["min_values"] = dict(cf.get("min_values") or {})

            # exception branch: same base constraints, but GPU filter is ONLY the exception GPU
            exc_branch: dict[str, Any] = {}
            if "include_values" in base_branch:
                exc_branch["include_values"] = dict(base_branch.get("include_values") or {})
            if "numeric_ranges" in base_branch:
                exc_branch["numeric_ranges"] = dict(base_branch.get("numeric_ranges") or {})
            if "min_values" in base_branch:
                exc_branch["min_values"] = dict(base_branch.get("min_values") or {})
            exc_branch["include_contains"] = {"gpu": [any2]}

            cf["any_of"] = [base_branch, exc_branch]
            cf.pop("include_values", None)
            cf.pop("include_contains", None)
            cf.pop("numeric_ranges", None)
            cf.pop("min_values", None)
        else:
            cf.pop("any_of", None)
    else:
        cf.pop("any_of", None)

    # remove unsupported keys that can confuse users (engine ignores them)
    cf.pop("all_of", None)

    obj["candidate_filter"] = cf
    return obj


def _apply_policy_create_ui_overrides(
    *, base_obj: dict[str, Any], ui_project: str, ui_platforms: list[str], strict: bool = True
) -> dict[str, Any]:
    """
    create UI의 입력값을 base_obj(보통 OpenAI/사용자 YAML) 위에 "최종값"으로 덮어쓴다.
    - 목적: UI에서 체크/입력한 값(특히 architecture, GPU 예외 any_of)이 YAML에 빠지거나 LLM에 의해 덮어써지는 문제 방지
    """
    obj: dict[str, Any] = dict(base_obj or {})
    obj["project"] = str(obj.get("project") or ui_project or "NEWPROJ").strip()
    obj["version"] = str(obj.get("version") or "v2").strip() or "v2"
    ui_desc = str(st.session_state.get("ui_policy_desc_create") or "").strip()
    if ui_desc:
        obj["description"] = ui_desc

    cf = obj.get("candidate_filter")
    if not isinstance(cf, dict):
        cf = {}
        obj["candidate_filter"] = cf

    # platform: always follow current UI selection (if empty, keep existing)
    plats = ui_platforms if ui_platforms else ["android", "ios"]
    cf["platform"] = ",".join([str(x).strip() for x in plats if str(x).strip()]) or str(cf.get("platform") or "android,ios")

    # include_values
    iv = cf.get("include_values")
    if not isinstance(iv, dict):
        iv = {}
        cf["include_values"] = iv

    # manufacturer (exact match list)
    manu_txt = str(st.session_state.get("ui_cf_manufacturer_create") or "").strip()
    manu = _unique_preserve_order(_split_csv_list(manu_txt))
    if manu:
        iv["manufacturer"] = [x.upper() for x in manu]
    else:
        if strict:
            iv.pop("manufacturer", None)

    # architecture: if exactly one selected, enforce it. If both/none => no restriction
    arch_sel: list[str] = []
    if bool(st.session_state.get("ui_cf_arch_64_create")):
        arch_sel.append("64bit")
    if bool(st.session_state.get("ui_cf_arch_32_create")):
        arch_sel.append("32bit")
    if len(arch_sel) == 1:
        iv["architecture"] = [arch_sel[0]]
    else:
        if strict:
            iv.pop("architecture", None)

    if not iv:
        # strict 모드에서는 UI가 비어있을 때 필터를 제거(=UI가 source of truth)
        # prompt 모드(strict=False)에서는 불필요한 빈 dict만 제거
        cf.pop("include_values", None)

    # include_contains.gpu (basic)
    ic = cf.get("include_contains")
    if not isinstance(ic, dict):
        ic = {}
        cf["include_contains"] = ic
    gpu_basic = str(st.session_state.get("ui_cf_gpu_contains_create") or "").strip()
    if gpu_basic:
        ic["gpu"] = [gpu_basic]
    else:
        if strict:
            ic.pop("gpu", None)
    # remove invalid include_contains.architecture if present (LLM mistake)
    ic.pop("architecture", None)
    if not ic:
        cf.pop("include_contains", None)

    # numeric_ranges.ram_gb
    if bool(st.session_state.get("ui_cf_ram_rng_on_create")):
        lo = float(st.session_state.get("ui_cf_ram_lo_create", 2) or 2)
        hi = float(st.session_state.get("ui_cf_ram_hi_create", 3) or 3)
        if lo > hi:
            lo, hi = hi, lo
        nr = cf.get("numeric_ranges")
        if not isinstance(nr, dict):
            nr = {}
            cf["numeric_ranges"] = nr
        nr["ram_gb"] = [lo, hi]
    else:
        if strict:
            nr = cf.get("numeric_ranges")
            if isinstance(nr, dict):
                nr.pop("ram_gb", None)
                if not nr:
                    cf.pop("numeric_ranges", None)

    # min_values.os_ver
    if bool(st.session_state.get("ui_cf_os_min_on_create")):
        os_min_txt = str(st.session_state.get("ui_cf_os_min_text_create") or "").strip()
        try:
            os_min_val = float(os_min_txt) if os_min_txt else None
        except Exception:
            os_min_val = None
        if os_min_val is not None:
            mv = cf.get("min_values")
            if not isinstance(mv, dict):
                mv = {}
                cf["min_values"] = mv
            mv["os_ver"] = os_min_val
    else:
        if strict:
            mv = cf.get("min_values")
            if isinstance(mv, dict):
                mv.pop("os_ver", None)
                if not mv:
                    cf.pop("min_values", None)

    # any_of: 예외 GPU(추가 확인용)
    # 요구 확정:
    # - 예외 GPU도 기본 조건(architecture/RAM/OS/제조사 등)은 반드시 만족해야 함
    # - 예외 목록은 "예외 GPU만" 대상으로 보여줌(기본 GPU 필터는 예외 목록에 영향 주지 않음)
    if bool(st.session_state.get("ui_cf_gpu_or_on_create")):
        any2_gpu = str(st.session_state.get("ui_cf_any2_gpu_create") or "").strip()
        if any2_gpu:
            # base branch: current AND filters (including basic GPU filter)
            base_branch: dict[str, Any] = {}
            iv0 = cf.get("include_values") if isinstance(cf.get("include_values"), dict) else {}
            ic0 = cf.get("include_contains") if isinstance(cf.get("include_contains"), dict) else {}
            nr0 = cf.get("numeric_ranges") if isinstance(cf.get("numeric_ranges"), dict) else {}
            mv0 = cf.get("min_values") if isinstance(cf.get("min_values"), dict) else {}
            if iv0:
                base_branch["include_values"] = dict(iv0)
            if ic0:
                base_branch["include_contains"] = dict(ic0)
            if nr0:
                base_branch["numeric_ranges"] = dict(nr0)
            if mv0:
                base_branch["min_values"] = dict(mv0)

            # exception branch: same base constraints, but GPU filter is ONLY the exception GPU
            exc_branch: dict[str, Any] = {}
            if iv0:
                exc_branch["include_values"] = dict(iv0)
            if nr0:
                exc_branch["numeric_ranges"] = dict(nr0)
            if mv0:
                exc_branch["min_values"] = dict(mv0)
            exc_branch["include_contains"] = {"gpu": [any2_gpu]}

            cf["any_of"] = [base_branch, exc_branch]

            # keep root clean (the engine evaluates any_of branches on top of base AND;
            # having the same constraints at root can cause confusion in logs/YAML)
            cf.pop("include_values", None)
            cf.pop("include_contains", None)
            cf.pop("numeric_ranges", None)
            cf.pop("min_values", None)
    else:
        # user didn't request exception GPU -> remove (strict 모드에서만)
        if strict:
            cf.pop("any_of", None)

    return obj


def _normalize_policy_obj_to_supported_v2(obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    OpenAI/legacy 포맷(candidate_filter.all_of 등)을 v2 엔진이 실제로 읽는 포맷으로 정규화한다.
    - 지원: include_values / include_contains / numeric_ranges / min_values / any_of / platform
    - all_of는 엔진에서 무시되므로, 가능하면 변환하고 최종적으로 제거한다.
    """
    warns: list[str] = []
    if not isinstance(obj, dict):
        return obj, ["YAML 최상위가 dict가 아닙니다."]

    # --- strip non-v2 top-level keys (prevent confusing YAML outputs) ---
    # These keys are not part of PolicyV2 schema and are ignored by loader/engine.
    for k in ("policy_name", "name", "platforms"):
        if k in obj:
            obj.pop(k, None)
            warns.append(f"최상위 '{k}'는 v2 스키마가 아니어서 제거했습니다.")

    # --- normalize max_per_product_name location ---
    # Canonical form in this app: dedupe.within_rank.max_per_product_name
    # Some outputs/older YAMLs may contain a top-level max_per_product_name; move it into dedupe.within_rank.
    top_mppn = obj.get("max_per_product_name")
    if top_mppn is not None:
        try:
            mppn_int = int(float(top_mppn))
        except Exception:
            mppn_int = None
        if mppn_int is not None:
            ded = obj.get("dedupe")
            if not isinstance(ded, dict):
                ded = {}
            dw = ded.get("within_rank")
            if not isinstance(dw, dict):
                dw = {}
            if dw.get("max_per_product_name") is None:
                dw["max_per_product_name"] = mppn_int
            ded["within_rank"] = dw
            obj["dedupe"] = ded
            warns.append("max_per_product_name은 dedupe.within_rank.max_per_product_name로 이동해 정규화했습니다.")
        # remove top-level key to avoid confusion/duplication
        obj.pop("max_per_product_name", None)
    cf = obj.get("candidate_filter")
    if not isinstance(cf, dict):
        return obj, warns

    def _ensure_list_str(v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        return [s] if s else []

    def _normalize_manufacturer_tokens(vals: list[str]) -> list[str]:
        """
        제조사 토큰을 엔진/데이터에서 흔히 쓰는 표준(대문자 영문)으로 정규화한다.
        - OpenAI/사용자가 한글로 입력한 경우가 있어 alias를 적용한다.
        """
        alias = {
            "삼성": "SAMSUNG",
            "삼성전자": "SAMSUNG",
            "애플": "APPLE",
            "구글": "GOOGLE",
            "엘지": "LG",
            "lg": "LG",
            "샤오미": "XIAOMI",
            "화웨이": "HUAWEI",
            "모토로라": "MOTOROLA",
            "레노버": "LENOVO",
            "소니": "SONY",
        }
        out: list[str] = []
        seen: set[str] = set()
        for raw in vals:
            s = str(raw or "").strip()
            if not s:
                continue
            key = s.strip()
            mapped = alias.get(key, None)
            if mapped:
                u = mapped
            else:
                # best-effort: already english -> upper
                u = s.upper()
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out

    def _clean_include_values(iv: dict[str, Any], *, ctx: str) -> dict[str, Any]:
        """
        LLM이 자주 섞어 넣는 '혼동 키'를 제거한다.
        - CandidateFilter에는 platform 필드가 별도로 있고, 엔진은 include_values.platform을 의미있게 쓰지 않는다.
        - include_values.os에 iOS/Android 같은 값이 들어오는 것도 혼동이므로 제거한다.
        """
        if not isinstance(iv, dict) or not iv:
            return iv
        if "platform" in iv:
            iv.pop("platform", None)
            warns.append(f"{ctx}: include_values.platform은 혼동을 유발해 제거했습니다.")
        osv = iv.get("os")
        if osv is not None:
            vals = [x.lower() for x in _ensure_list_str(osv)]
            if any(v in ("ios", "android", "aos") for v in vals):
                iv.pop("os", None)
                warns.append(f"{ctx}: include_values.os(iOS/Android)는 혼동을 유발해 제거했습니다.")

        manu = iv.get("manufacturer")
        if manu is not None:
            norm = _normalize_manufacturer_tokens(_ensure_list_str(manu))
            if norm:
                if norm != list(_ensure_list_str(manu)):
                    warns.append(f"{ctx}: include_values.manufacturer를 표준 표기(대문자/alias)로 정규화했습니다.")
                iv["manufacturer"] = norm
            else:
                iv.pop("manufacturer", None)
        return iv

    def _normalize_any_of_branches(cf_dict: dict[str, Any]) -> None:
        """
        any_of 분기 구조가 깨진(OpenAI 출력) 케이스를 best-effort로 복구한다.
        예) - gpu: {include_contains: [PowerVR]}  ->  - include_contains: {gpu: [PowerVR]}
        """
        anyv = cf_dict.get("any_of")
        if not isinstance(anyv, list) or not anyv:
            return
        def _norm_numeric_ranges(d: dict[str, Any]) -> None:
            nr = d.get("numeric_ranges")
            if not isinstance(nr, dict):
                return
            for k, v in list(nr.items()):
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        nr[str(k)] = [float(v[0]), float(v[1])]
                    except Exception:
                        continue
            d["numeric_ranges"] = nr

        fixed: list[dict[str, Any]] = []
        for br in anyv:
            if not isinstance(br, dict):
                continue
            b = dict(br)
            # platform sometimes appears as dict; drop it (platform should be string at root)
            if "platform" in b and not isinstance(b.get("platform"), (str, int, float)):
                b.pop("platform", None)
            # broken gpu nesting
            gv = b.get("gpu")
            if isinstance(gv, dict) and "include_contains" in gv and "include_contains" not in b:
                b["include_contains"] = {"gpu": _ensure_list_str(gv.get("include_contains"))}
                b.pop("gpu", None)
                warns.append("candidate_filter.any_of: 'gpu: {include_contains: ...}' 포맷을 include_contains.gpu로 정규화했습니다.")
            # if include_contains.gpu is a string, coerce to list for consistency
            icb = b.get("include_contains")
            if isinstance(icb, dict) and "gpu" in icb:
                icb["gpu"] = _ensure_list_str(icb.get("gpu"))
                b["include_contains"] = icb
            # normalize numeric_ranges numbers for stable dedupe
            _norm_numeric_ranges(b)
            fixed.append(b)
        # de-dup branches (OpenAI often outputs redundant branches, or base promotion creates near-duplicates)
        if fixed:
            uniq: list[dict[str, Any]] = []
            seen = set()
            for b in fixed:
                try:
                    sig = yaml.safe_dump(b, allow_unicode=True, sort_keys=True)
                except Exception:
                    sig = str(b)
                if sig in seen:
                    continue
                seen.add(sig)
                uniq.append(b)
            cf_dict["any_of"] = uniq

    def _move_contains_manufacturer_to_values(d: dict[str, Any], *, ctx: str) -> None:
        """
        manufacturer는 'contains'로 넣으면 의도와 달라질 수 있으므로(부분 문자열),
        include_contains.manufacturer가 있으면 include_values.manufacturer로 이동한다.
        (예: APPLE만 -> 정확히 APPLE만)
        """
        if not isinstance(d, dict):
            return
        ic0 = d.get("include_contains")
        if not isinstance(ic0, dict) or "manufacturer" not in ic0:
            return
        manu_vals = _ensure_list_str(ic0.get("manufacturer"))
        ic0.pop("manufacturer", None)
        if not ic0:
            d.pop("include_contains", None)
        else:
            d["include_contains"] = ic0

        if manu_vals:
            iv0 = d.get("include_values")
            if not isinstance(iv0, dict):
                iv0 = {}
            norm = _normalize_manufacturer_tokens(manu_vals)
            if norm:
                iv0["manufacturer"] = norm
                d["include_values"] = iv0
                warns.append(f"{ctx}: include_contains.manufacturer → include_values.manufacturer 로 정규화했습니다.")

    def _promote_base_filters_into_any_of(cf_dict: dict[str, Any]) -> None:
        """
        예외 OR(any_of)을 'A ∪ B'로 해석할 때, root(AND)에 있는 base 조건들을 any_of에 흡수한다.
        - 단, root에 RAM 범위 같은 '약한 조건'만 남아있는 경우는 새 OR 브랜치로 만들면 결과가 과도하게 넓어지므로
          이미 존재하는 'base 분기(제조사/OS/아키텍처 등)'에 병합한다. (예시10 방지)
        """
        anyv = cf_dict.get("any_of")
        if not isinstance(anyv, list) or not anyv:
            return

        base_branch: dict[str, Any] = {}
        ivx = cf_dict.get("include_values")
        icx = cf_dict.get("include_contains")
        nrx = cf_dict.get("numeric_ranges")
        mvx = cf_dict.get("min_values")
        if isinstance(ivx, dict) and ivx:
            base_branch["include_values"] = ivx
        if isinstance(icx, dict) and icx:
            base_branch["include_contains"] = icx
        if isinstance(nrx, dict) and nrx:
            base_branch["numeric_ranges"] = nrx
        if isinstance(mvx, dict) and mvx:
            base_branch["min_values"] = mvx

        if not base_branch:
            return

        # remove root filters
        cf_dict.pop("include_values", None)
        cf_dict.pop("include_contains", None)
        cf_dict.pop("numeric_ranges", None)
        cf_dict.pop("min_values", None)

        # If base_branch is "weak-only" (no include_values/include_contains), merge into an existing strong branch if possible.
        weak_only = ("include_values" not in base_branch) and ("include_contains" not in base_branch)
        if weak_only:
            target_i: int | None = None
            for i, br in enumerate(anyv):
                if not isinstance(br, dict):
                    continue
                # prefer a branch that already has include_values (manufacturer/architecture) or os_ver
                if isinstance(br.get("include_values"), dict) and br.get("include_values"):
                    target_i = i
                    break
                mvb = br.get("min_values")
                if isinstance(mvb, dict) and ("os_ver" in mvb):
                    target_i = i
                    break
            if target_i is not None:
                br = dict(anyv[target_i]) if isinstance(anyv[target_i], dict) else {}
                # deep-merge only missing keys inside numeric_ranges/min_values
                for k in ("numeric_ranges", "min_values"):
                    src = base_branch.get(k)
                    if not isinstance(src, dict) or not src:
                        continue
                    dst = br.get(k)
                    if not isinstance(dst, dict):
                        dst = {}
                    for kk, vv in src.items():
                        if kk not in dst:
                            dst[kk] = vv
                    br[k] = dst
                anyv[target_i] = br
                warns.append("any_of(예외 OR) 보정: root의 약한 조건(RAM/OS 등)을 base 분기에 병합했습니다.")
                cf_dict["any_of"] = anyv
                _normalize_any_of_branches(cf_dict)
                return

        # default: prepend as a base branch
        cf_dict["any_of"] = [base_branch, *anyv]
        warns.append("any_of(예외 OR) 의도에 맞게, base 조건을 any_of[0]로 승격해 A∪B 형태로 정규화했습니다.")
        _normalize_any_of_branches(cf_dict)

        # additional cleanup happens in _strip_duplicate_constraints_from_gpu_branches()

    def _strip_duplicate_constraints_from_gpu_branches(cf_dict: dict[str, Any]) -> None:
        """
        예외 OR(any_of)에서 사용자가 원하는 경우가 많음:
        - base 조건(A) ∪ GPU 예외(B: GPU만으로 전체 포함)
        그런데 LLM/정규화 과정에서 B 분기에 base의 RAM/OS 같은 제약이 '같이' 붙는 경우가 있음.
        -> base(any_of[0])와 동일한 numeric_ranges/min_values가 GPU 분기에 붙어 있으면 제거한다.
        """
        anyv = cf_dict.get("any_of")
        if not isinstance(anyv, list) or len(anyv) < 2:
            return
        base0 = anyv[0] if isinstance(anyv[0], dict) else {}
        base_nr = base0.get("numeric_ranges") if isinstance(base0.get("numeric_ranges"), dict) else {}
        base_mv = base0.get("min_values") if isinstance(base0.get("min_values"), dict) else {}
        if not isinstance(base_nr, dict):
            base_nr = {}
        if not isinstance(base_mv, dict):
            base_mv = {}
        for i in range(1, len(anyv)):
            br = anyv[i]
            if not isinstance(br, dict):
                continue
            icb = br.get("include_contains")
            if not (isinstance(icb, dict) and icb.get("gpu")):
                continue
            # Only strip when this is a "GPU-only exception branch" (intended to break base conditions).
            # If the branch already carries base constraints (include_values/numeric_ranges/min_values),
            # we keep them because some workflows require "exception GPU must still satisfy base conditions".
            has_iv = isinstance(br.get("include_values"), dict) and bool(br.get("include_values"))
            has_nr = isinstance(br.get("numeric_ranges"), dict) and bool(br.get("numeric_ranges"))
            has_mv = isinstance(br.get("min_values"), dict) and bool(br.get("min_values"))
            if has_iv or has_nr or has_mv:
                continue
            nr_b = br.get("numeric_ranges") if isinstance(br.get("numeric_ranges"), dict) else None
            mv_b = br.get("min_values") if isinstance(br.get("min_values"), dict) else None
            if isinstance(nr_b, dict) and base_nr and nr_b == base_nr:
                br.pop("numeric_ranges", None)
                warns.append("any_of(예외 OR) 보정: GPU 예외 분기에서 base와 동일한 numeric_ranges를 제거했습니다.")
            if isinstance(mv_b, dict) and base_mv and mv_b == base_mv:
                br.pop("min_values", None)
                warns.append("any_of(예외 OR) 보정: GPU 예외 분기에서 base와 동일한 min_values를 제거했습니다.")
            anyv[i] = br
        cf_dict["any_of"] = anyv
        _normalize_any_of_branches(cf_dict)

    def _normalize_min_values_keys_in_cf(cf_dict: dict[str, Any], *, ctx: str) -> None:
        """
        LLM이 자주 틀리는 min_values key를 엔진이 실제로 쓰는 컬럼명으로 정규화한다.
        - 정책 엔진은 min_values의 key를 DataFrame 컬럼명으로 그대로 사용하므로, 잘못된 키는 조건이 '조용히' 무시될 수 있다.
        """
        mv = cf_dict.get("min_values")
        if not isinstance(mv, dict) or not mv:
            return
        # os_version -> os_ver
        if "os_version" in mv and "os_ver" not in mv:
            mv["os_ver"] = mv.get("os_version")
            mv.pop("os_version", None)
            warns.append(f"{ctx}: min_values.os_version → min_values.os_ver 로 정규화했습니다.")
        # common alias variants
        if "osver" in mv and "os_ver" not in mv:
            mv["os_ver"] = mv.get("osver")
            mv.pop("osver", None)
            warns.append(f"{ctx}: min_values.osver → min_values.os_ver 로 정규화했습니다.")
        if "os" in mv and "os_ver" not in mv:
            # 너무 공격적일 수 있어 숫자만 들어오는 케이스에 한해 보정
            try:
                _ = float(mv.get("os"))  # type: ignore[arg-type]
                mv["os_ver"] = mv.get("os")
                mv.pop("os", None)
                warns.append(f"{ctx}: min_values.os → min_values.os_ver 로 정규화했습니다.")
            except Exception:
                pass
        cf_dict["min_values"] = mv

    # Fix common LLM mistake: platform rendered as dict (invalid). Replace with a sensible default.
    if "platform" in cf and not isinstance(cf.get("platform"), (str, int, float)):
        cf.pop("platform", None)
        warns.append("candidate_filter.platform가 문자열이 아니어서 제거했습니다(기본 플랫폼은 UI/휴리스틱으로 채워집니다).")
    # Normalize platform string tokens (best-effort)
    if isinstance(cf.get("platform"), (str, int, float)):
        plat = str(cf.get("platform") or "").strip().lower()
        if plat:
            plat = plat.replace("|", ",").replace(";", ",").replace(" ", "")
            cf["platform"] = plat

    # normalize min_values keys (root + any_of branches)
    _normalize_min_values_keys_in_cf(cf, ctx="candidate_filter")
    any0 = cf.get("any_of")
    if isinstance(any0, list):
        for i, br in enumerate(any0):
            if isinstance(br, dict):
                _normalize_min_values_keys_in_cf(br, ctx=f"candidate_filter.any_of[{i}]")

    # normalize any_of branch shapes early
    _normalize_any_of_branches(cf)
    # fix common manufacturer mistake (root + any_of branches)
    _move_contains_manufacturer_to_values(cf, ctx="candidate_filter")
    any1 = cf.get("any_of")
    if isinstance(any1, list):
        for i, br in enumerate(any1):
            if isinstance(br, dict):
                _move_contains_manufacturer_to_values(br, ctx=f"candidate_filter.any_of[{i}]")
    # after manufacturer fixes, strip unintended duplicate constraints from GPU exception branches
    _strip_duplicate_constraints_from_gpu_branches(cf)

    all_of = cf.get("all_of")
    if not isinstance(all_of, list):
        # still normalize include_values best-effort
        iv0 = cf.get("include_values")
        if isinstance(iv0, dict):
            # manufacturer casing + de-dup
            manu = iv0.get("manufacturer")
            if isinstance(manu, (list, tuple)):
                norm = []
                seen = set()
                for x in manu:
                    s = str(x).strip()
                    if not s:
                        continue
                    u = s.upper()
                    if u in seen:
                        continue
                    seen.add(u)
                    norm.append(u)
                if norm and norm != list(manu):
                    iv0["manufacturer"] = norm
                    warns.append("include_values.manufacturer를 대문자/중복제거로 정규화했습니다.")
            # cpu_arch -> architecture (if architecture missing)
            if "cpu_arch" in iv0 and "architecture" not in iv0:
                iv0["architecture"] = iv0.get("cpu_arch")
                warns.append("include_values.cpu_arch를 include_values.architecture로 이동했습니다.")
            # drop cpu_arch to avoid confusion (engine may not have such column)
            if "cpu_arch" in iv0:
                iv0.pop("cpu_arch", None)
            cf["include_values"] = _clean_include_values(iv0, ctx="candidate_filter")

        # normalize numeric_ranges best-effort (LLM often outputs {min,max} dicts)
        nr0 = cf.get("numeric_ranges")
        if isinstance(nr0, dict) and nr0:
            changed = False
            for k, v in list(nr0.items()):
                if isinstance(v, dict) and ("min" in v and "max" in v):
                    try:
                        nr0[str(k)] = [float(v["min"]), float(v["max"])]
                        changed = True
                    except Exception:
                        continue
            if changed:
                cf["numeric_ranges"] = nr0
                warns.append("numeric_ranges의 {min,max} 포맷을 [min,max] 리스트 포맷으로 정규화했습니다.")

        # any_of A∪B semantics tuning (non-all_of path)
        _promote_base_filters_into_any_of(cf)
        _strip_duplicate_constraints_from_gpu_branches(cf)

        obj["candidate_filter"] = cf
        return obj, warns

    # containers
    iv = cf.get("include_values") if isinstance(cf.get("include_values"), dict) else {}
    ic = cf.get("include_contains") if isinstance(cf.get("include_contains"), dict) else {}
    nr = cf.get("numeric_ranges") if isinstance(cf.get("numeric_ranges"), dict) else {}
    mv = cf.get("min_values") if isinstance(cf.get("min_values"), dict) else {}

    def _ensure_list(v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        return [s] if s else []

    for cond in all_of:
        if not isinstance(cond, dict):
            continue

        # style 1: {"field":"ram_gb","operator":"between","min":1,"max":3}
        if "field" in cond and "operator" in cond:
            field = str(cond.get("field") or "").strip()
            op = str(cond.get("operator") or "").strip().lower()
            if not field:
                continue
            if op in ("equals", "eq"):
                val = str(cond.get("value") or "").strip()
                if val:
                    iv[field] = [val]
            elif op in ("between",):
                try:
                    lo = float(cond.get("min"))
                    hi = float(cond.get("max"))
                    nr[field] = [lo, hi]
                except Exception:
                    warns.append(f"between 변환 실패: {cond}")
            elif op in ("contains",):
                val = str(cond.get("value") or "").strip()
                if val:
                    ic[field] = [val]
            else:
                warns.append(f"지원하지 않는 operator: {op} ({cond})")
            continue

        # style 2: {"ram_gb":{"between":[1,3]}} or {"arch":{"in":["x86_64","arm64"]}}
        for k, v in cond.items():
            kk = str(k).strip()
            if not kk:
                continue
            if isinstance(v, dict) and "between" in v:
                rng = v.get("between")
                if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                    try:
                        nr[kk] = [float(rng[0]), float(rng[1])]
                    except Exception:
                        warns.append(f"between 변환 실패: {cond}")
                continue
            if isinstance(v, dict) and "in" in v:
                vals = _ensure_list(v.get("in"))
                if vals:
                    iv[kk] = vals
                continue
            if isinstance(v, dict) and "equals" in v:
                val = str(v.get("equals") or "").strip()
                if val:
                    iv[kk] = [val]
                continue

    if iv:
        # normalize manufacturer casing + de-dup
        manu = iv.get("manufacturer")
        if isinstance(manu, (list, tuple)):
            norm = []
            seen = set()
            for x in manu:
                s = str(x).strip()
                if not s:
                    continue
                u = s.upper()
                if u in seen:
                    continue
                seen.add(u)
                norm.append(u)
            if norm and norm != list(manu):
                iv["manufacturer"] = norm
                warns.append("include_values.manufacturer를 대문자/중복제거로 정규화했습니다.")
        # cpu_arch -> architecture (if architecture missing)
        if "cpu_arch" in iv and "architecture" not in iv:
            iv["architecture"] = iv.get("cpu_arch")
            warns.append("include_values.cpu_arch를 include_values.architecture로 이동했습니다.")
        # drop cpu_arch to avoid confusion
        if "cpu_arch" in iv:
            iv.pop("cpu_arch", None)
        cf["include_values"] = _clean_include_values(iv, ctx="candidate_filter")
    if ic:
        # cpu_arch in include_contains is ambiguous / usually not a real column → drop to avoid confusion
        if "cpu_arch" in ic:
            ic.pop("cpu_arch", None)
            warns.append("include_contains.cpu_arch는 모호하거나 컬럼이 없는 경우가 많아 자동 제거했습니다.")
        cf["include_contains"] = ic
    if nr:
        # normalize numeric_ranges.ram_gb if written as {min,max}
        rr = nr.get("ram_gb")
        if isinstance(rr, dict) and ("min" in rr and "max" in rr):
            try:
                nr["ram_gb"] = [float(rr["min"]), float(rr["max"])]
                warns.append("numeric_ranges.ram_gb를 [min,max] 리스트 포맷으로 정규화했습니다.")
            except Exception:
                pass
        # common mistake: only "min" provided -> treat as exact [min,min]
        rr2 = nr.get("ram_gb")
        if isinstance(rr2, dict) and ("min" in rr2 and "max" not in rr2):
            try:
                v = float(rr2["min"])
                nr["ram_gb"] = [v, v]
                warns.append("numeric_ranges.ram_gb에 min만 있어 [min,min]으로 정규화했습니다.")
            except Exception:
                pass
        # conflict guard: if numeric_ranges.ram_gb exists, drop min_values.ram_gb
        mv0 = cf.get("min_values")
        if isinstance(mv0, dict) and ("ram_gb" in mv0):
            mv0.pop("ram_gb", None)
            if not mv0:
                cf.pop("min_values", None)
            else:
                cf["min_values"] = mv0
            warns.append("numeric_ranges.ram_gb가 지정되어 min_values.ram_gb는 충돌 방지를 위해 제거했습니다.")
        cf["numeric_ranges"] = nr
    if mv:
        cf["min_values"] = mv

    # any_of A∪B semantics tuning (all_of path after containers assembled)
    _promote_base_filters_into_any_of(cf)
    _strip_duplicate_constraints_from_gpu_branches(cf)

    # remove unsupported all_of to prevent "looks applied but ignored"
    cf.pop("all_of", None)
    obj["candidate_filter"] = cf
    warns.append("candidate_filter.all_of가 감지되어 지원 포맷(include_values/numeric_ranges 등)으로 정규화했습니다.")
    return obj, warns


def _heuristic_fill_missing_filters_from_text(
    obj: dict[str, Any],
    *,
    requirement_text: str,
    default_project: str | None = None,
    default_platforms: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    LLM이 조건을 누락하는 케이스(예: 'RAM 1~3GB'를 말했는데 numeric_ranges가 비어있음)를 보완한다.
    - 현재 UI/프롬프트에서 자주 쓰는 패턴만 best-effort로 지원한다.
    """
    warns: list[str] = []
    if not isinstance(obj, dict):
        return obj, warns
    txt = str(requirement_text or "")
    # 사용자가 "64bit만_RAM_4GB_이하" 처럼 언더스코어로 구분해서 쓰는 케이스가 많아
    # 휴리스틱 파싱을 위해 공백으로 정규화한다.
    txt_norm = txt.replace("_", " ")
    low = txt_norm.lower()

    # ensure required keys
    if default_project and not str(obj.get("project") or "").strip():
        obj["project"] = str(default_project).strip()
        warns.append("project가 없어 기본 프로젝트 값을 자동 주입했습니다.")
    if not str(obj.get("version") or "").strip():
        obj["version"] = "v2"

    cf = obj.get("candidate_filter")
    if not isinstance(cf, dict):
        cf = {}
        obj["candidate_filter"] = cf

    # platform hint (프롬프트에 platform이 명시된 경우, OpenAI 출력이 틀려도 프롬프트를 우선한다)
    has_android = any(tok in low for tok in ["aos", "android", "안드로이드"])
    has_ios = any(tok in low for tok in ["ios", "아이폰", "ipad"])
    if any(tok in low for tok in ["android만", "android 만", "안드로이드만", "안드로이드 만"]):
        cf["platform"] = "android"
        warns.append("요구사항에서 Android만을 감지해 platform=android로 보정했습니다.")
    elif any(tok in low for tok in ["ios만", "ios 만", "아이폰만", "아이폰 만", "ipad만", "ipad 만"]):
        cf["platform"] = "ios"
        warns.append("요구사항에서 iOS만을 감지해 platform=ios로 보정했습니다.")
    elif has_android and has_ios:
        cf["platform"] = "android,ios"
        warns.append("요구사항에서 Android+iOS를 감지해 platform=android,ios로 보정했습니다.")
    elif has_android and not has_ios:
        cf["platform"] = "android"
        warns.append("요구사항에서 Android/AOS를 감지해 platform=android로 보정했습니다.")
    elif has_ios and not has_android:
        cf["platform"] = "ios"
        warns.append("요구사항에서 iOS를 감지해 platform=ios로 보정했습니다.")

    # platform default (also fix invalid/non-string platform values)
    if default_platforms:
        plat = cf.get("platform")
        if not isinstance(plat, (str, int, float)) or (not str(plat).strip()):
            cf["platform"] = ",".join(default_platforms)
            warns.append("platform이 비어있거나 잘못된 형식이라 기본 플랫폼(android,ios)을 적용했습니다.")

    # RAM intent parsing (프롬프트가 명시한 의도를 OpenAI 출력보다 우선):
    # 우선순위(중요): "범위(1~3[만])" > "만(정확히)" > "이상(>=)"
    nr = cf.get("numeric_ranges")
    if not isinstance(nr, dict):
        nr = {}
    mv = cf.get("min_values")
    if not isinstance(mv, dict):
        mv = {}

    # 1) range: "1~3GB" (optionally ends with "만")
    # NOTE: "1~3GB만"은 '3GB만'이 아니라 '1~3GB만' 의미다 → range가 최우선
    m_rng = (
        re.search(r"(?:ram|램)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[\~\-]\s*(\d+(?:\.\d+)?)\s*gb(?:\s*만)?", low)
        or re.search(r"(\d+(?:\.\d+)?)\s*[\~\-]\s*(\d+(?:\.\d+)?)\s*gb(?:\s*만)?", low)
    )
    if m_rng:
        try:
            lo = float(m_rng.group(1))
            hi = float(m_rng.group(2))
            if lo > hi:
                lo, hi = hi, lo
            nr["ram_gb"] = [lo, hi]
            mv.pop("ram_gb", None)
            warns.append(f"요구사항에서 RAM 범위({lo:g}~{hi:g}GB)를 감지해 numeric_ranges.ram_gb=[{lo:g},{hi:g}]로 보정했습니다.")
        except Exception:
            pass

    # 2) exact-only: "... 4GB만" (range가 아닐 때만)
    if not m_rng:
        m_exact = re.search(r"(?:ram|램)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*gb\s*만", low) or re.search(
            r"\b(\d+(?:\.\d+)?)\s*gb\s*만\b", low
        )
        if m_exact:
            try:
                v = float(m_exact.group(1))
                nr["ram_gb"] = [v, v]
                mv.pop("ram_gb", None)
                warns.append(f"요구사항에서 RAM {v:g}GB만(정확히)을 감지해 numeric_ranges.ram_gb=[{v:g},{v:g}]로 보정했습니다.")
            except Exception:
                pass

    # 3) min: "3GB 이상"
    if (not m_rng) and ("ram_gb" not in nr):
        m_min = re.search(r"(?:ram|램)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*gb\s*(?:이상|over|>=)", low) or re.search(
            r"\b(\d+(?:\.\d+)?)\s*gb\s*(?:이상|over|>=)\b", low
        )
        if m_min:
            try:
                v = float(m_min.group(1))
                mv["ram_gb"] = v
                nr.pop("ram_gb", None)
                warns.append(f"요구사항에서 RAM {v:g}GB 이상(>=)을 감지해 min_values.ram_gb={v:g}로 보정했습니다.")
            except Exception:
                pass

    # 4) max: "4GB 이하" (<=) — range/exact/min 보다 약하지만, 프롬프트에 자주 등장한다.
    if (not m_rng) and ("ram_gb" not in nr):
        m_max = re.search(r"(?:ram|램)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*gb\s*(?:이하|under|<=)", low) or re.search(
            r"\b(\d+(?:\.\d+)?)\s*gb\s*(?:이하|under|<=)\b", low
        )
        if m_max:
            try:
                v = float(m_max.group(1))
                nr["ram_gb"] = [0.0, v]
                mv.pop("ram_gb", None)
                warns.append(f"요구사항에서 RAM {v:g}GB 이하(<=)를 감지해 numeric_ranges.ram_gb=[0,{v:g}]로 보정했습니다.")
            except Exception:
                pass

    if nr:
        cf["numeric_ranges"] = nr
    else:
        cf.pop("numeric_ranges", None)
    if mv:
        cf["min_values"] = mv
    else:
        cf.pop("min_values", None)

    # architecture 64bit hint
    # NOTE: 32bit/64bit "만" 의도는 강하게 반영한다.
    if "32bit" in low or "32-bit" in low:
        iv = cf.get("include_values")
        if not isinstance(iv, dict):
            iv = {}
            cf["include_values"] = iv
        iv["architecture"] = ["32bit"]
        warns.append("요구사항에서 32bit를 감지해 include_values.architecture=[32bit]로 보정했습니다.")
    elif "64bit" in low or "64-bit" in low:
        iv = cf.get("include_values")
        if not isinstance(iv, dict):
            iv = {}
            cf["include_values"] = iv
        iv["architecture"] = ["64bit"]
        warns.append("요구사항에서 64bit를 감지해 include_values.architecture=[64bit]로 보정했습니다.")

    obj["candidate_filter"] = cf
    return obj, warns


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


def _postprocess_output_for_platform(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    플랫폼별 결과 표/엑셀 출력용 후처리.
    - iOS: 아키텍처는 데이터/의미가 없거나 비어 있는 경우가 많아 결과에서 제거
    """
    if df is None or df.empty:
        return df
    p = str(platform or "").strip().lower()
    if p == "ios":
        return df.drop(columns=["아키텍처"], errors="ignore")
    return df


def _exception_gpu_matches_df(*, master_df: pd.DataFrame, policy: "PolicyV2") -> tuple[pd.DataFrame, list[str]]:
    """
    예외 GPU(any_of 중 include_contains.gpu 분기)로 매칭되는 '전체 목록'을 반환한다.
    - 선정(Selected)과 별개로, 후보 필터 단계와 동일한 로직(platform/available 포함)으로 계산한다.
    - 중복 제거/랭크 타겟/제조사 분산은 적용하지 않는다.
    """
    try:
        from app.policy_v2 import PolicyV2, CandidateFilter, normalize_df_for_policy_v2
        from app.policy_v2_engine import _apply_candidate_filter  # type: ignore
    except Exception:
        return (master_df.head(0).copy() if master_df is not None else pd.DataFrame(), [])

    cf0 = getattr(policy, "candidate_filter", None)
    if cf0 is None:
        return (master_df.head(0).copy() if master_df is not None else pd.DataFrame(), [])

    toks: list[str] = []
    try:
        for br in tuple(getattr(cf0, "any_of", None) or ()):
            ic = getattr(br, "include_contains", None) or {}
            if isinstance(ic, dict) and ic.get("gpu"):
                for x in (ic.get("gpu") or ()):
                    s = str(x).strip()
                    if s and s not in toks:
                        toks.append(s)
    except Exception:
        toks = []

    if not toks:
        return (master_df.head(0).copy() if master_df is not None else pd.DataFrame(), [])

    # "예외 GPU 전체"도 UI에서 설정한 기본 조건(64bit/OS/제조사/RAM 등)은 반드시 만족해야 한다.
    # - 정책이 A∪B 형태(any_of)라면, 기본 조건은 보통 any_of[0]에 들어간다.
    # - 그렇지 않으면 root의 AND 조건(include_values/min_values/numeric_ranges/include_contains 등)을 사용한다.
    base_iv: dict[str, tuple[str, ...]] = {}
    base_mv: dict[str, float] = {}
    base_nr: dict[str, tuple[float, float]] = {}
    base_ic: dict[str, tuple[str, ...]] = {}

    def _pick_base_branch() -> object | None:
        anyv = tuple(getattr(cf0, "any_of", None) or ())
        if not anyv:
            return None
        # prefer a "non-gpu-only" branch as base
        for br in anyv:
            try:
                ic = getattr(br, "include_contains", None) or {}
                iv = getattr(br, "include_values", None) or {}
                mv = getattr(br, "min_values", None) or {}
                nr = getattr(br, "numeric_ranges", None) or {}
                has_gpu = isinstance(ic, dict) and bool(ic.get("gpu"))
                has_other = (isinstance(iv, dict) and bool(iv)) or (isinstance(mv, dict) and bool(mv)) or (isinstance(nr, dict) and bool(nr))
                if has_other and not (has_gpu and not has_other):
                    return br
            except Exception:
                continue
        # fallback: first branch
        for br in anyv:
            return br
        return None

    base_br = _pick_base_branch()
    src = base_br if base_br is not None else cf0

    try:
        iv0 = getattr(src, "include_values", None) or {}
        if isinstance(iv0, dict):
            base_iv = {str(k): tuple(str(x).strip() for x in (v or ()) if str(x).strip()) for k, v in iv0.items() if str(k).strip()}
    except Exception:
        base_iv = {}
    try:
        mv0 = getattr(src, "min_values", None) or {}
        if isinstance(mv0, dict):
            for k, v in mv0.items():
                kk = str(k).strip()
                if not kk:
                    continue
                try:
                    base_mv[kk] = float(v)
                except Exception:
                    continue
    except Exception:
        base_mv = {}
    try:
        nr0 = getattr(src, "numeric_ranges", None) or {}
        if isinstance(nr0, dict):
            for k, v in nr0.items():
                kk = str(k).strip()
                if not kk:
                    continue
                try:
                    lo = float(v[0])
                    hi = float(v[1])
                    base_nr[kk] = (lo, hi)
                except Exception:
                    continue
    except Exception:
        base_nr = {}
    try:
        ic0 = getattr(src, "include_contains", None) or {}
        if isinstance(ic0, dict):
            # NOTE: 예외 목록은 "예외 GPU"만 대상으로 하므로, root/base의 gpu 필터는 합치지 않는다.
            for k, v in ic0.items():
                kk = str(k).strip()
                if not kk or kk == "gpu":
                    continue
                base_ic[kk] = tuple(str(x).strip() for x in (v or ()) if str(x).strip())
    except Exception:
        base_ic = {}

    # AND 필터로 "기본 조건 + 예외 GPU"를 동시에 만족하는 전체 목록을 만든다.
    base_cf = CandidateFilter(
        platform=str(getattr(cf0, "platform", "") or ""),
        required_fields=tuple(getattr(cf0, "required_fields", None) or ()),
        include_values=base_iv,
        include_contains={**base_ic, "gpu": tuple(toks)},
        min_values=base_mv,
        numeric_ranges=base_nr,
        any_of=(),
        availability=getattr(cf0, "availability", None),
    )
    pol2 = PolicyV2(
        project=str(getattr(policy, "project", "") or "EXCEPTION_GPU"),
        version=str(getattr(policy, "version", "") or "v2"),
        candidate_filter=base_cf,
        dedupe=getattr(policy, "dedupe", None),
        diversity=getattr(policy, "diversity", None),
        tie_breakers=tuple(getattr(policy, "tie_breakers", None) or ()),
        versioning=getattr(policy, "versioning", None),
        focus=getattr(policy, "focus", None),
        manufacturer_policy=getattr(policy, "manufacturer_policy", None),
    )
    # IMPORTANT:
    # - master_df는 normalize_testbed() 결과로, v2 엔진이 기대하는 표준 컬럼(architecture/ram_gb 등)이 없을 수 있다.
    # - 예외 GPU 테이블도 선정(run_policy_v2)와 동일한 기준으로 계산해야 하므로,
    #   v2 표준 컬럼으로 한 번 더 정규화한 DF에 후보 필터를 적용한다.
    master_df_v2 = normalize_df_for_policy_v2(master_df)

    logs: list[str] = []
    try:
        out = _apply_candidate_filter(master_df_v2, pol2, logs)
    except Exception:
        out = master_df_v2.head(0).copy() if master_df_v2 is not None else pd.DataFrame()

    # 예외 목록 후처리:
    # - (cpu/gpu/ram_gb) 스펙 중복은 제거하지 않음
    # - 대신 product_name(또는 '제품명') 기준으로 1개만 남김
    # - 정렬: No 접두어 A > IR > R (동일 우선순위 내 원래 순서 유지)
    if out is not None and not out.empty:
        out = out.copy()
        out["_orig_i"] = range(len(out))

        # sort by No prefix priority
        no_col = "No" if "No" in out.columns else ("NO" if "NO" in out.columns else "")
        if no_col:
            s_no = out[no_col].fillna("").astype(str).str.strip().str.upper()
            # 예외 목록 정렬 우선순위: A > AT > R
            # NOTE: "AT"는 "A"로도 시작하므로, AT를 먼저 체크해야 의도대로 분리된다.
            def _pri(x: str) -> int:
                if x.startswith("A") and (not x.startswith("AT")):
                    return 0
                if x.startswith("AT"):
                    return 1
                if x.startswith("R"):
                    return 2
                return 999

            out["_no_pri_exc"] = s_no.map(_pri)
            out = out.sort_values(by=["_no_pri_exc", "_orig_i"], ascending=[True, True])

        # product_name dedupe (keep first after sorting)
        pn_col = "product_name" if "product_name" in out.columns else ("제품명" if "제품명" in out.columns else "")
        if pn_col:
            pn = out[pn_col].fillna("").astype(str).str.strip().str.upper()
            m_nonempty = pn.ne("")
            out_non = out[m_nonempty].copy()
            pn_non = pn[m_nonempty]
            out_non = out_non.loc[~pn_non.duplicated(keep="first")].copy()
            out_empty = out[~m_nonempty].copy()
            out = pd.concat([out_non, out_empty], axis=0, ignore_index=True, sort=False)

            # preserve ordering after concat (use existing sort cols if present)
            if "_no_pri_exc" in out.columns and "_orig_i" in out.columns:
                out = out.sort_values(by=["_no_pri_exc", "_orig_i"], ascending=[True, True])

        out = out.drop(columns=["_orig_i", "_no_pri_exc"], errors="ignore")

    return out, toks


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
        # 정책 필터는 candidate_filter.include_values.architecture(표준 컬럼)를 사용하므로,
        # 출력도 표준 architecture를 우선 반영하고, 없을 때만 원본 '아키텍처'로 fallback 한다.
        "아키텍처": ["architecture", "아키텍처", "arch"],
        "OS Ver": ["OS Ver", "OSVer", "os_ver"],
        "선정 사유": ["why_short", "why", "why_detail"],
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

    def _best_reason_col(cols: list[str]) -> str | None:
        """
        '선정 사유'는 why_short/why/why_detail 중에서 실제 값이 가장 많이 채워진 컬럼을 선택한다.
        (캐시/조인/구버전 혼선 등으로 why_short가 비어 있으면 why/why_detail로 자동 fallback)
        """
        best: str | None = None
        best_cnt = -1
        for c in cols:
            got = c if c in df.columns else None
            if not got:
                k = _norm_key(c)
                got_list = norm_to_cols.get(k) or []
                got = _best_non_empty(got_list) if got_list else None
            if not got or got not in df.columns:
                continue
            try:
                s = df[got].fillna("").astype(str).str.strip()
                cnt = int(s.ne("").sum())
            except Exception:
                cnt = 0
            if cnt > best_cnt:
                best_cnt = cnt
                best = got
        return best if best_cnt > 0 else (_first_existing(cols) or None)

    out = pd.DataFrame(index=df.index)
    for out_name in out_cols:
        if out_name == "선정 사유":
            src = _best_reason_col(mapping.get(out_name, []))
        else:
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
        cols_keep = [c for c in ["device_id", "why_short", "why", "why_detail"] if c in dl.columns]
        if not cols_keep:
            return sel
        dl2 = dl[cols_keep].copy()
        out = sel.merge(dl2.drop_duplicates(subset=["device_id"]), on="device_id", how="left")
    elif "No" in sel.columns and "No" in dl.columns:
        cols_keep = [c for c in ["No", "why_short", "why", "why_detail"] if c in dl.columns]
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


def _policy_local_summary(yaml_text: str, *, policy_filename: str | None = None) -> str:
    """
    OpenAI 없이도 볼 수 있는 정책 해석(로컬).
    - '정책 해석' 버튼에서 항상 동일한 포맷으로 출력하기 위해 사용한다.
    """

    def _as_str_list(v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v if str(x).strip() != ""]
        # allow "android,ios" string
        s = str(v).strip()
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s] if s else []

    def _fmt_list(xs: list[str]) -> str:
        return ", ".join(xs) if xs else "-"

    try:
        obj = yaml.safe_load(yaml_text or "")
    except Exception as e:
        return f"YAML 파싱 실패: {e}"
    if not isinstance(obj, dict):
        return "정책 YAML 최상위가 dict가 아닙니다."

    proj = str(obj.get("project") or "").strip()
    proj_desc = {
        "KP": "Korea Publishing",
        "KRJP": "Korea / Japan",
        "PALM": "PALM",
    }.get(proj, "")
    fn = (policy_filename or (f"{proj}.yaml" if proj else "")).strip()
    title = f"{fn} ({proj_desc})".strip()

    cf = obj.get("candidate_filter") or {}
    availability = (cf.get("availability") or {}) if isinstance(cf, dict) else {}
    platform = _as_str_list(cf.get("platform") if isinstance(cf, dict) else "")
    required_fields = _as_str_list(cf.get("required_fields") if isinstance(cf, dict) else "")
    usable_only = bool(availability.get("usable_only")) if isinstance(availability, dict) else False
    rentable_only = bool(availability.get("rentable_only")) if isinstance(availability, dict) else False

    dedupe = obj.get("dedupe") or {}
    dw = (dedupe.get("within_rank") or {}) if isinstance(dedupe, dict) else {}
    dedupe_key = _as_str_list(dw.get("key") if isinstance(dw, dict) else "")
    max_per_product_name = dw.get("max_per_product_name") if isinstance(dw, dict) else None
    allow_dupes_if_diff = _as_str_list(dw.get("allow_duplicates_if_differs_by") if isinstance(dw, dict) else "")
    allow_duplicates = bool(dw.get("allow_duplicates")) if isinstance(dw, dict) else False
    max_dupes_per_profile = dw.get("max_duplicates_per_profile") if isinstance(dw, dict) else None

    diversity = obj.get("diversity") or {}
    vw = (diversity.get("within_rank") or {}) if isinstance(diversity, dict) else {}
    must_cover = _as_str_list(vw.get("must_cover") if isinstance(vw, dict) else "")
    optional_axes = _as_str_list(vw.get("optional_axes") if isinstance(vw, dict) else "")

    focus = obj.get("focus") or {}
    preferred_ranks = _as_str_list((focus.get("preferred_ranks") if isinstance(focus, dict) else "") or "")

    manu = obj.get("manufacturer_policy") or {}
    mw = (manu.get("within_rank") or {}) if isinstance(manu, dict) else {}
    manu_mode = str(mw.get("mode") or "").strip() if isinstance(mw, dict) else ""
    manu_weight = mw.get("penalty_weight") if isinstance(mw, dict) else None

    tb = obj.get("tie_breakers") or []
    tie_breaker_lines: list[str] = []
    if isinstance(tb, (list, tuple)):
        for t in tb:
            if isinstance(t, str):
                tie_breaker_lines.append(t)
                continue
            if isinstance(t, dict) and len(t) == 1:
                k = next(iter(t.keys()))
                v = t.get(k)
                if k == "no_prefix_priority" and isinstance(v, dict):
                    a = _as_str_list(v.get("android"))
                    i = _as_str_list(v.get("ios"))
                    tie_breaker_lines.append(
                        f"no_prefix_priority (android: {' > '.join(a) if a else '-'}, ios: {' > '.join(i) if i else '-'})"
                    )
                elif k == "gpu_family_priority" and isinstance(v, (list, tuple)):
                    tie_breaker_lines.append(f"gpu_family_priority=[{', '.join([str(x) for x in v])}]")
                else:
                    tie_breaker_lines.append(f"{k}: {v}")
            else:
                tie_breaker_lines.append(str(t))

    axis_ko = {
        "display_bucket": "디스플레이 해상도/비율",
        "form_factor": "폼팩터(Phone/Tablet)",
        "target_market": "타겟 마켓",
    }
    allow_diff_ko = [axis_ko.get(x, x) for x in allow_dupes_if_diff]

    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(f"대상 플랫폼: {_fmt_list(platform)}" + (" (AOS/iOS 각각 분리 실행)" if len(platform) > 1 else ""))
    lines.append(f"후보 필터: usable_only: {str(usable_only).lower()} (상태가 사용가능/CQA만), rentable_only: {str(rentable_only).lower()}")
    lines.append(f"필수 컬럼: {_fmt_list(required_fields)}")
    lines.append("중복/대표성(dedupe):")
    lines.append(f"  기본 대표 프로파일: key=[{', '.join(dedupe_key)}]" if dedupe_key else "  기본 대표 프로파일: -")
    if max_per_product_name is not None:
        lines.append(f"  제품명 중복 제한: max_per_product_name: {max_per_product_name} (Rank 내 제품명 1대)")
    if allow_dupes_if_diff:
        lines.append(f"  제한적 중복 허용(커버리지 확장): allow_duplicates_if_differs_by=[{', '.join(allow_dupes_if_diff)}]")
        lines.append(f"  같은 CPU/GPU/RAM이어도 {_fmt_list(allow_diff_ko)} 다르면 추가로 뽑힐 수 있음")
    if allow_duplicates:
        lines.append(f"  중복 허용: allow_duplicates: true (max_duplicates_per_profile: {max_dupes_per_profile})")

    lines.append("다양성(diversity):")
    if must_cover:
        lines.append(f"  must_cover=[{', '.join(must_cover)}] → 같은 Rank에서 다양성 확보를 유도")
    else:
        lines.append("  must_cover=-")
    if optional_axes:
        lines.append(f"  optional_axes=[{', '.join(optional_axes)}] → “가능하면” 다양하게 커버하도록 점수/우선순위에 반영")
    else:
        lines.append("  optional_axes=-")

    if preferred_ranks:
        lines.append(f"포커스(focus): preferred_ranks=[{', '.join(preferred_ranks)}]")

    if manu_mode:
        lines.append(
            f"제조사 정책: {manu_mode}"
            + (f" + penalty_weight: {manu_weight}" if manu_weight is not None else "")
            + " (같은 Rank에서 특정 제조사 쏠림을 완화)"
        )

    if tie_breaker_lines:
        lines.append("타이브레이커:")
        for x in tie_breaker_lines:
            lines.append(f"  - {x}")
    return "\n".join(lines)


def _policy_local_summary_simple(yaml_text: str) -> str:
    """
    비개발자도 이해할 수 있는 1문단 요약.
    - 카드/리스트에서 짧게 보여주기 용도
    """

    def _as_str_list(v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v if str(x).strip() != ""]
        s = str(v).strip()
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s] if s else []

    def _join(xs: list[str]) -> str:
        return ", ".join([x for x in xs if str(x).strip()]) if xs else ""

    def _fmt_range(v: object) -> str:
        # expects dict like {min: 2, max: 3} or list/tuple like [2,3]
        if isinstance(v, dict):
            lo = v.get("min")
            hi = v.get("max")
            if lo is not None and hi is not None:
                return f"{lo}~{hi}"
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return f"{v[0]}~{v[1]}"
        return ""

    try:
        obj = yaml.safe_load(yaml_text or "")
    except Exception:
        return "YAML을 해석하지 못했습니다."
    if not isinstance(obj, dict):
        return "정책 형식이 올바르지 않습니다."

    cf = obj.get("candidate_filter") or {}
    if not isinstance(cf, dict):
        cf = {}
    availability = cf.get("availability") or {}
    if not isinstance(availability, dict):
        availability = {}

    plats = _as_str_list(cf.get("platform"))
    plat_txt = " / ".join([p.upper() if p in ("ios", "android") else str(p) for p in plats]) if plats else ""
    if set(plats) == {"android", "ios"}:
        plat_txt = "Android / iOS"
    elif plats == ["android"]:
        plat_txt = "Android"
    elif plats == ["ios"]:
        plat_txt = "iOS"

    usable_only = bool(availability.get("usable_only"))
    base_filters: list[str] = []
    if plat_txt:
        base_filters.append(plat_txt)
    if usable_only:
        base_filters.append("사용 가능한 기기(CQA 포함)")

    include_values = cf.get("include_values") or {}
    include_contains = cf.get("include_contains") or {}
    min_values = cf.get("min_values") or {}
    numeric_ranges = cf.get("numeric_ranges") or {}
    if not isinstance(include_values, dict):
        include_values = {}
    if not isinstance(include_contains, dict):
        include_contains = {}
    if not isinstance(min_values, dict):
        min_values = {}
    if not isinstance(numeric_ranges, dict):
        numeric_ranges = {}

    extras: list[str] = []
    manu = _as_str_list(include_values.get("manufacturer"))
    if manu:
        extras.append(f"제조사: {_join(manu)}")
    arch = _as_str_list(include_values.get("architecture"))
    if arch:
        extras.append(f"아키텍처: {_join(arch)}")
    gpu_contains = _as_str_list(include_contains.get("gpu"))
    if gpu_contains:
        extras.append(f"GPU에 '{_join(gpu_contains)}' 포함")

    ram_rng = numeric_ranges.get("ram_gb")
    ram_rng_txt = _fmt_range(ram_rng)
    if ram_rng_txt:
        extras.append(f"RAM: {ram_rng_txt}GB")
    os_min = min_values.get("os_ver")
    if os_min is not None and str(os_min).strip() != "":
        extras.append(f"OS: {os_min} 이상")

    # OR/예외(any_of)
    any_of = cf.get("any_of") or []
    any_parts: list[str] = []
    if isinstance(any_of, (list, tuple)) and any_of:
        for br in any_of:
            if not isinstance(br, dict):
                continue
            br_iv = br.get("include_values") or {}
            br_ic = br.get("include_contains") or {}
            br_min = br.get("min_values") or {}
            br_rng = br.get("numeric_ranges") or {}
            if not isinstance(br_iv, dict):
                br_iv = {}
            if not isinstance(br_ic, dict):
                br_ic = {}
            if not isinstance(br_min, dict):
                br_min = {}
            if not isinstance(br_rng, dict):
                br_rng = {}
            one: list[str] = []
            br_ram_min = br_min.get("ram_gb")
            if br_ram_min is not None and str(br_ram_min).strip() != "":
                one.append(f"RAM {br_ram_min}GB 이상")
            br_gpu = _as_str_list(br_ic.get("gpu"))
            if br_gpu:
                one.append(f"GPU에 '{_join(br_gpu)}' 포함")
            br_os = br_min.get("os_ver")
            if br_os is not None and str(br_os).strip() != "":
                one.append(f"OS {br_os} 이상")
            br_ram_rng = _fmt_range(br_rng.get("ram_gb"))
            if br_ram_rng:
                one.append(f"RAM {br_ram_rng}GB")
            if one:
                any_parts.append(" / ".join(one))

    filter_sentence = ""
    if base_filters:
        filter_sentence = " / ".join(base_filters) + "만 후보로 선별됩니다."
    if extras:
        filter_sentence = (filter_sentence[:-1] + f" + ({', '.join(extras)}) 조건을 적용합니다.") if filter_sentence else f"{', '.join(extras)} 조건으로 후보를 선별합니다."
    if any_parts:
        filter_sentence += f" 예외(OR): {' 또는 '.join(any_parts)}"

    # dedupe/diversity
    dedupe = obj.get("dedupe") or {}
    dw = (dedupe.get("within_rank") or {}) if isinstance(dedupe, dict) else {}
    max_per_product_name = dw.get("max_per_product_name") if isinstance(dw, dict) else None
    allow_dupes_if_diff = _as_str_list(dw.get("allow_duplicates_if_differs_by") if isinstance(dw, dict) else "")
    dedupe_key = _as_str_list(dw.get("key") if isinstance(dw, dict) else "")
    if not dedupe_key:
        dedupe_key = ["cpu", "gpu", "ram_gb"]

    diversity = obj.get("diversity") or {}
    vw = (diversity.get("within_rank") or {}) if isinstance(diversity, dict) else {}
    must_cover = _as_str_list(vw.get("must_cover") if isinstance(vw, dict) else "")

    dedupe_sentence_parts: list[str] = []
    if max_per_product_name is not None:
        dedupe_sentence_parts.append("같은 제품명은 1대만")
    if dedupe_key:
        if set(dedupe_key) >= {"cpu", "gpu", "ram_gb"}:
            dedupe_sentence_parts.append("같은 스펙(CPU+GPU+RAM)은 대표 1대만")
        else:
            dedupe_sentence_parts.append(f"같은 스펙({'+'.join(dedupe_key)})은 대표 1대만")
    if allow_dupes_if_diff:
        dedupe_sentence_parts.append("필요 시 사용 경험(UX)이 다른 경우만 제한적으로 추가")
    dedupe_sentence = ""
    if dedupe_sentence_parts:
        dedupe_sentence = "중복은 " + ", ".join(dedupe_sentence_parts) + " 남기도록 줄입니다."

    diversity_sentence = ""
    if "cpu_family" in must_cover:
        diversity_sentence = "또한 CPU 종류가 골고루 나오도록 선정합니다."

    out = " ".join([x for x in [filter_sentence, dedupe_sentence, diversity_sentence] if x]).strip()
    return out or "정책 요약을 생성할 수 없습니다."


def _safe_policy_filename(stem: str) -> str:
    s = _safe_str(stem)
    if not s:
        return ""
    safe = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", ".", " "))
    safe = safe.strip().replace(" ", "_")
    safe = safe.replace(".yaml", "").replace(".yml", "")
    return safe


def _toast_once(message: str, *, icon: str | None = None) -> None:
    """
    Show a toast once after rerun.
    - Streamlit toast is ephemeral and best for "저장 완료" 알림.
    """
    msg = _safe_str(message)
    if not msg:
        return
    st.session_state["_toast_once_msg"] = msg
    if icon is not None:
        st.session_state["_toast_once_icon"] = str(icon)


def _render_toast_once() -> None:
    msg = _safe_str(st.session_state.pop("_toast_once_msg", ""))
    if not msg:
        return
    icon = _safe_str(st.session_state.pop("_toast_once_icon", ""))
    # st.toast may not exist on older Streamlit versions
    toast_fn = getattr(st, "toast", None)
    if callable(toast_fn):
        try:
            toast_fn(msg, icon=icon if icon else None)
            return
        except Exception:
            pass
    # fallback
    st.success(msg)


def _policy_editor_page(*, policy_files: list[Path], selected_policy: Path) -> None:
    """
    정책 확인/수정 페이지
    """
    from app.policy_v2 import load_policy_v2

    # policy_screen: catalog | view | edit | create
    st.session_state.setdefault("policy_screen", "catalog")
    _render_toast_once()

    # --- URL query params: restore screen/pick on refresh (best-effort) ---
    # 목적: 새로고침 시 policy_screen/policy_editor_pick이 초기화되어 YAML이 "다른 정책"으로 바뀌어 보이는 문제 방지
    def _qp_get() -> dict[str, str]:
        try:
            qp = getattr(st, "query_params", None)
            if qp is not None:
                return {str(k): str(qp.get(k, "")) for k in list(qp.keys())}
        except Exception:
            pass
        # legacy fallback
        try:
            fn = getattr(st, "experimental_get_query_params", None)
            if callable(fn):
                raw = fn() or {}
                out: dict[str, str] = {}
                for k, v in (raw.items() if isinstance(raw, dict) else []):
                    if isinstance(v, (list, tuple)) and v:
                        out[str(k)] = str(v[0])
                    else:
                        out[str(k)] = str(v or "")
                return out
        except Exception:
            pass
        return {}

    def _qp_set(params: dict[str, str]) -> None:
        try:
            qp = getattr(st, "query_params", None)
            if qp is not None:
                try:
                    qp.clear()
                except Exception:
                    pass
                qp.update({str(k): str(v) for k, v in (params or {}).items()})
                return
        except Exception:
            pass
        # legacy fallback
        try:
            fn2 = getattr(st, "experimental_set_query_params", None)
            if callable(fn2):
                fn2(**{str(k): str(v) for k, v in (params or {}).items()})
        except Exception:
            pass

    q = _qp_get()
    q_screen = str(q.get("policy_screen") or "").strip().lower()
    q_pick = str(q.get("policy") or "").strip()
    if q_screen in ("catalog", "view", "edit", "create"):
        st.session_state["policy_screen"] = q_screen

    # cached card data
    @st.cache_data(show_spinner=False)
    def _policy_card_data(path_str: str, mtime_ns: int) -> dict[str, str]:
        p = Path(path_str)
        raw = p.read_text(encoding="utf-8")
        policy_name = p.stem  # file stem (.yaml 없이)
        project = ""
        desc = ""
        try:
            obj = yaml.safe_load(raw) or {}
            if isinstance(obj, dict):
                project = str(obj.get("project") or "").strip()
                desc = str(obj.get("description") or "")
        except Exception:
            pass

        friendly = {
            "KP": "KP (Korea Publishing)",
            "KRJP": "KRJP (Korea-Japan)",
            "PALM": "PALM",
        }
        # "프로젝트 정책 이름"은 파일명으로 통일한다. (project는 별도 표시)
        display_name = policy_name
        project_label = friendly.get(project, project) if project else ""
        if not desc:
            default_desc = {
                "KP": "한국 퍼블리싱 프로젝트 정책",
                "KRJP": "한국-일본 통합 프로젝트 정책",
                "PALM": "PALM 프로젝트 정책",
            }
            desc = default_desc.get(project, "") if project else ""

        summary = _policy_local_summary(raw, policy_filename=p.name)
        simple = _policy_local_summary_simple(raw)
        return {
            "display_name": display_name,
            "project": project,
            "project_label": project_label,
            "desc": desc,
            "summary": summary,
            "simple": simple,
            "raw": raw,
        }

    # top bar (polished / product-like)
    screen = str(st.session_state.get("policy_screen") or "catalog")

    def _reset_create_session_state() -> None:
        """
        '새 정책 만들기'에 진입할 때마다 create 화면 입력값만 초기화한다.
        - API 설정/상태(api_status, show_api_dialog, OPENAI_API_KEY 등)는 건드리지 않는다.
        """
        keys = [
            # create form inputs
            "ui_policy_name_create",
            "ui_policy_desc_create",
            "ui_cf_manufacturer_create",
            "ui_cf_arch_64_create",
            "ui_cf_arch_32_create",
            "ui_cf_gpu_contains_create",
            "ui_cf_ram_rng_on_create",
            "ui_cf_ram_lo_create",
            "ui_cf_ram_hi_create",
            "ui_cf_os_min_on_create",
            "ui_cf_os_min_text_create",
            "ui_cf_gpu_or_on_create",
            "ui_cf_any2_gpu_create",
            # create text/yaml
            "policy_natural_text_create",
            "policy_create_yaml_text",
            "new_policy_name_create",
            # debug results (optional)
            "example10_results",
        ]
        for k in keys:
            st.session_state.pop(k, None)

    # create 화면으로 "진입"할 때마다(create -> create rerun은 제외) 입력값을 초기화한다.
    prev_screen = str(st.session_state.get("_policy_screen_prev") or "")
    if screen == "create" and prev_screen != "create":
        _reset_create_session_state()
    st.session_state["_policy_screen_prev"] = screen
    if screen == "create":
        title = "새 정책 만들기"
        subtitle = "새 정책을 만들고 저장합니다. (기존 정책 파일은 변경되지 않습니다)"
    elif screen == "edit":
        title = "정책 수정하기"
        subtitle = "선택한 정책 YAML을 수정하고 검증/저장할 수 있습니다."
    elif screen == "view":
        title = "정책 상세 보기"
        subtitle = "정책 내용을 읽기 전용으로 확인합니다."
    else:
        title = "정책 관리"
        subtitle = "등록된 정책을 확인/편집하거나 새 정책을 생성할 수 있습니다."

    top_l, top_c, top_r = st.columns([2, 6, 2])
    with top_l:
        if st.button("추출 화면으로 이동하기", key="btn_policy_top_go_main", use_container_width=True):
            st.session_state["page_mode"] = "run"
            # policy editor 상태 query param 제거(메인 화면에는 불필요)
            _qp_set({})
            st.rerun()

        # catalog 화면에서만 '새 정책 만들기'를 노출한다(편집/상세/생성 화면에서는 중복/혼동 방지)
        if screen == "catalog":
            if st.button("새 정책 만들기", key="btn_policy_top_new_left", use_container_width=True, type="primary"):
                _reset_create_session_state()
                st.session_state["policy_screen"] = "create"
                # NOTE: st.rerun()은 즉시 실행을 중단하므로, query param도 여기서 함께 갱신해야
                # 다음 rerun 시작 시 q_screen 복원이 create로 유지된다.
                _qp_set({"policy_screen": "create", "policy": str(st.session_state.get("policy_editor_pick") or "")})
                st.rerun()

    with top_c:
        st.markdown(
            f"""
<div style="text-align:center; padding-top: 6px;">
  <div style="font-size:26px; font-weight:900; letter-spacing:-0.02em;">{title}</div>
  <div style="margin-top:4px; font-size:13px; color: rgba(17,24,39,0.65);">{subtitle}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with top_r:
        # 우측 액션: 제거(요청 사항: 정책 관리 화면 우측 '새 정책' 버튼 삭제)
        pass

    # Determine current policy (for view/edit)
    policy_names = [p.stem for p in policy_files]
    # prefer query param pick on refresh
    if q_pick and (q_pick in policy_names):
        st.session_state["policy_editor_pick"] = q_pick

    pick = st.session_state.get("policy_editor_pick") or selected_policy.stem
    if pick not in policy_names:
        pick = selected_policy.stem if selected_policy.stem in policy_names else (policy_names[0] if policy_names else "")
    st.session_state["policy_editor_pick"] = pick
    policy_path = next(p for p in policy_files if p.stem == pick) if pick else selected_policy
    raw_text = policy_path.read_text(encoding="utf-8") if policy_path and policy_path.exists() else ""

    # --- shared template lines (for create) ---
    prompt_template_lines = [
        "너는 'QA 디바이스 추천 v2 정책'을 YAML로 작성하는 도우미다.",
        "출력은 반드시 YAML만(설명/주석/코드블럭 금지).",
        "",
        "[값만 채우기 템플릿(비전공자용)]",
        "- 프로젝트: [KP|KRJP|PALM]",
        "- 플랫폼: [android|ios|android,ios]",
        "",
        "[후보 필터(선택, 없으면 빈칸)]",
        "",
        "[1) 정확히 일치(include_values) — 값이 '완전히 같은' 것만 통과]",
        "- 제조사(manufacturer): [예: SAMSUNG, APPLE] / [없음]",
        "- 아키텍처(architecture): [예: 64bit] / [없음]",
        "",
        "[2) 부분 포함(include_contains) — 문자열에 '포함'되면 통과(대소문자 무시)]",
        "- GPU 문자열(gpu): [예: PowerVR] / [없음]",
        "",
        "[3) 숫자 범위(numeric_ranges) — min~max 범위만 통과]",
        "- 램(GB)(ram_gb): [예: 2~3] / [없음]",
        "- 출시년도(release_year): [예: 2023~2026] / [없음]",
        "",
        "[4) 최소값(min_values) — 이상(>=)만 통과]",
        "- 출시년도 최소(release_year>=): [예: 2024] / [없음]",
        "- OS 버전 최소(os_ver>=): [예: 15] / [없음]",
        "",
        "[5) OR/예외(any_of) — 아래 조건 중 '하나라도' 만족하면 포함(합집합)]",
        "- 예: (RAM>=3) OR (GPU에 PowerVR 포함) 처럼 OR이 필요하면 any_of를 사용",
        "  - any_of 예시:",
        "    candidate_filter:",
        "      platform: android,ios",
        "      any_of:",
        "        - min_values: {ram_gb: 3}",
        "        - include_contains: {gpu: [powervr]}",
        "",
        "[다양성(골고루 뽑기) 규칙]",
        "- GPU 종류 다양하게(gpu_family): [예/아니오] (선택)",
        "",
        "[제조사 쏠림 완화]",
        "- 같은 Rank에서 한 제조사가 너무 많이 뽑히지 않게: [기본 적용(필수)]",
        "- 완화 강도(0.0~1.0): [0.5 고정]",
        "",
        "[주의: YAML이 아닌 UI에서만 설정되는 것(정책 항목 아님)]",
        "- 제외할 타겟 국가(멀티셀렉트), 제외할 NOTE(멀티셀렉트)",
        "- 필수 디바이스 No, 직전 버전 제외 No",
        "",
    ]

    # EDIT state (per policy)
    st.session_state.setdefault("policy_edit_yaml_text", raw_text)
    if st.session_state.get("_policy_edit_last_pick") != pick:
        st.session_state["_policy_edit_last_pick"] = pick
        st.session_state["policy_edit_yaml_text"] = raw_text
        st.session_state.pop("policy_edit_interpretation", None)
        st.session_state.pop("policy_view_interpretation", None)

    # CREATE state (independent)
    st.session_state.setdefault("policy_create_yaml_text", "")
    st.session_state.setdefault("policy_natural_text_create", "")
    # create YAML source tracking:
    # - "ui": UI 입력 기반(요구사항 문장 자동 생성 등) → 저장 시 UI가 source of truth
    # - "prompt": 프롬프트(직접 입력) 기반 → 저장 시 YAML을 그대로 보존
    st.session_state.setdefault("_policy_create_yaml_source", "ui")

    screen = str(st.session_state.get("policy_screen") or "catalog")
    # keep URL in sync with current state so refresh keeps same screen/pick
    if screen in ("catalog", "view", "edit", "create"):
        _qp_set({"policy_screen": screen, "policy": pick})

    if screen == "catalog":
        st.session_state.setdefault("policy_delete_confirm", "")
        for p in policy_files:
            mtime_ns = int(p.stat().st_mtime_ns)
            d = _policy_card_data(str(p), mtime_ns)
            with st.container(border=True):
                # Header row: policy name + actions inline
                # Put actions right next to the title (left-aligned), not on the far right.
                # 마지막 spacer 컬럼으로 남는 공간을 밀어내서 버튼이 제목 옆에 붙어 보이게 한다.
                h1, h2, h3, h4, h_sp = st.columns([6, 0.8, 0.8, 0.8, 10])
                with h1:
                    st.markdown(f"### {d['display_name']}")
                    # show project + description (project는 YAML의 project)
                    proj_txt = str(d.get("project_label") or d.get("project") or "").strip()
                    if proj_txt:
                        st.caption(f"프로젝트: {proj_txt}")
                    if d["desc"]:
                        st.caption(d["desc"])
                with h2:
                    if st.button("✏️", key=f"btn_edit_{p.stem}", help="수정", use_container_width=True):
                        st.session_state["policy_editor_pick"] = p.stem
                        st.session_state["policy_screen"] = "edit"
                        _qp_set({"policy_screen": "edit", "policy": p.stem})
                        st.rerun()
                with h3:
                    if st.button("📄", key=f"btn_view_{p.stem}", help="상세 보기", use_container_width=True):
                        st.session_state["policy_editor_pick"] = p.stem
                        st.session_state["policy_screen"] = "view"
                        _qp_set({"policy_screen": "view", "policy": p.stem})
                        st.rerun()
                with h4:
                    # delete (two-step confirm)
                    if st.button("🗑️", key=f"btn_del_{p.stem}", help="삭제", use_container_width=True):
                        st.session_state["policy_delete_confirm"] = p.stem
                        st.rerun()
                with h_sp:
                    pass

                if str(st.session_state.get("policy_delete_confirm") or "") == p.stem:
                    st.warning(f"정말 삭제할까요? `{p.name}` (되돌릴 수 없습니다)")
                    dc1, dc2 = st.columns([1, 1])
                    if dc1.button("삭제 확정", key=f"btn_del_confirm_{p.stem}", type="primary", use_container_width=True):
                        try:
                            p.unlink(missing_ok=True)
                            # if deleted policy was selected, clear selection best-effort
                            if st.session_state.get("policy_editor_pick") == p.stem:
                                st.session_state["policy_editor_pick"] = ""
                        finally:
                            st.session_state["policy_delete_confirm"] = ""
                        st.rerun()
                    if dc2.button("취소", key=f"btn_del_cancel_{p.stem}", use_container_width=True):
                        st.session_state["policy_delete_confirm"] = ""
                        st.rerun()

                st.markdown("**정책 요약(비개발자용)**")
                st.write(str(d.get("simple") or ""))
                with st.expander(f"YAML 해석 확인 ({p.stem})", expanded=False):
                    st.code(str(d["summary"]), language="text")
        return

    if screen == "view":
            st.subheader("정책 확인")
            st.caption("정책 내용을 읽기 전용으로 확인합니다.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("정책 해석(고정 포맷)", use_container_width=True):
                    ytxt = str(st.session_state.get("policy_edit_yaml_text", ""))
                    st.session_state["policy_view_interpretation"] = _policy_local_summary(
                        ytxt, policy_filename=policy_path.name
                    )
                if st.session_state.get("policy_view_interpretation"):
                    st.code(str(st.session_state.get("policy_view_interpretation") or ""), language="text")
            with c2:
                with st.expander("YAML 원문 보기", expanded=True):
                    st.code(str(st.session_state.get("policy_edit_yaml_text", "")), language="yaml")
            return

    if screen == "edit":
            # UI 기반 수정: 새 정책 만들기 화면과 동일한 폼 구성을 사용하고,
            # 선택된 정책 YAML을 파싱해 값이 자동으로 채워진 상태로 시작한다.
            st.session_state.setdefault("_policy_edit_ui_last_pick", "")
            st.session_state.setdefault("_policy_edit_just_saved", False)
            # Prefill when policy changes (prevents "꼬임" after edits/saves)
            need_prefill = st.session_state.get("_policy_edit_ui_last_pick") != pick
            # After save, never re-prefill (keeps user edits visible)
            if bool(st.session_state.get("_policy_edit_just_saved")):
                need_prefill = False
                st.session_state["_policy_edit_just_saved"] = False
            if need_prefill:
                try:
                    base_obj = yaml.safe_load(raw_text) or {}
                    if not isinstance(base_obj, dict):
                        base_obj = {}
                except Exception:
                    base_obj = {}
                _prefill_policy_edit_ui_from_yaml(obj=base_obj, policy_stem=policy_path.stem)
                st.session_state["_policy_edit_ui_last_pick"] = pick

            # base policy obj (from current YAML text if available, else raw)
            try:
                base_obj2 = yaml.safe_load(str(st.session_state.get("policy_edit_yaml_text") or raw_text)) or {}
                if not isinstance(base_obj2, dict):
                    base_obj2 = yaml.safe_load(raw_text) or {}
                if not isinstance(base_obj2, dict):
                    base_obj2 = {}
            except Exception:
                try:
                    base_obj2 = yaml.safe_load(raw_text) or {}
                    if not isinstance(base_obj2, dict):
                        base_obj2 = {}
                except Exception:
                    base_obj2 = {}

            # project는 YAML에 저장된 값을 우선 유지한다(정책 파일명과 혼동 방지)
            ui_project_for_edit = str(base_obj2.get("project") or "").strip() or "NEWPROJ"
            merged_obj = _apply_policy_edit_ui_to_obj(base_obj=base_obj2, ui_project=ui_project_for_edit)
            merged_yaml = yaml.safe_dump(merged_obj, allow_unicode=True, sort_keys=False)
            # YAML preview (read-only): always keep in sync with the UI
            st.session_state["policy_edit_yaml_preview"] = merged_yaml

            c1, c2 = st.columns(2)
            with c1:
                st.caption("선택한 정책의 값을 기반으로 자동 입력되어 있습니다. 필요한 항목만 수정하세요.")
                st.text_input(
                    "프로젝트 정책 이름 (필수)",
                    value=str(st.session_state.get("ui_policy_name_edit") or policy_path.stem),
                    key="ui_policy_name_edit",
                )
                st.text_input("설명", value=str(st.session_state.get("ui_policy_desc_edit") or ""), key="ui_policy_desc_edit")

                # 플랫폼(표시만): 정책 YAML에서 읽은 값
                plats = st.session_state.get("ui_cf_platforms_edit") or ["android", "ios"]
                st.caption(f"플랫폼: {', '.join([str(x) for x in plats])}")

                m1, m2 = st.columns(2)
                manu_txt = m1.text_input(
                    "제조사 필터(정확히 일치, 쉼표, 대소문자 무시)",
                    value=str(st.session_state.get("ui_cf_manufacturer_edit") or ""),
                    key="ui_cf_manufacturer_edit",
                    placeholder="예: SAMSUNG, APPLE",
                    help="선택한 제조사의 디바이스만 후보에 포함됩니다. 비워두면 모든 제조사를 포함합니다.",
                )
                with m2:
                    st.caption("아키텍처(정확히 일치, 선택)")
                    a1, a2 = st.columns(2)
                    with a1:
                        st.checkbox("64bit", value=bool(st.session_state.get("ui_cf_arch_64_edit", True)), key="ui_cf_arch_64_edit")
                    with a2:
                        st.checkbox("32bit", value=bool(st.session_state.get("ui_cf_arch_32_edit", True)), key="ui_cf_arch_32_edit")

                st.text_input(
                    "GPU 포함(기본 필터)",
                    value=str(st.session_state.get("ui_cf_gpu_contains_edit") or ""),
                    key="ui_cf_gpu_contains_edit",
                    placeholder="예: powervr / adreno",
                    help="“이 GPU가 들어간 기기만 보고 싶다” → 이 조건을 만족하는 것만 후보로 남깁니다. (대소문자 무시)",
                )

                n1, n2, n3 = st.columns(3)
                ram_rng_on = n1.checkbox("RAM 범위", value=bool(st.session_state.get("ui_cf_ram_rng_on_edit", False)), key="ui_cf_ram_rng_on_edit", help="예: 2~3GB")
                if ram_rng_on:
                    r1, r2 = n1.columns(2)
                    r1.number_input("최소(GB)", min_value=0, value=int(st.session_state.get("ui_cf_ram_lo_edit", 2) or 2), step=1, format="%d", key="ui_cf_ram_lo_edit")
                    r2.number_input("최대(GB)", min_value=0, value=int(st.session_state.get("ui_cf_ram_hi_edit", 3) or 3), step=1, format="%d", key="ui_cf_ram_hi_edit")
                    n1.caption("RAM(GB) 최소~최대 범위만 후보로 통과합니다.")

                os_min_on = n2.checkbox("OS 최소", value=bool(st.session_state.get("ui_cf_os_min_on_edit", False)), key="ui_cf_os_min_on_edit", help="예: Android 9+, iOS 15+")
                if os_min_on:
                    n2.text_input(
                        "OS 버전(이상)",
                        value=str(st.session_state.get("ui_cf_os_min_text_edit") or ""),
                        key="ui_cf_os_min_text_edit",
                        placeholder="예: 9 / 15 / 17.0.1",
                        help="예: 9 이상이면 Android 9+, iOS 15 이상이면 iOS 15+ 같은 의미",
                    )
                    n2.caption("입력한 버전 이상(>=)만 후보로 통과합니다.")

                gpu_or_on = n3.checkbox(
                    "GPU 포함(기본조건 만족 못해도 포함)",
                    value=bool(st.session_state.get("ui_cf_gpu_or_on_edit", False)),
                    key="ui_cf_gpu_or_on_edit",
                    help="기본 조건은 유지하되, 해당 GPU도 포함해서 보고싶다",
                )
                if gpu_or_on:
                    n3.text_input(
                        "GPU 포함 문자열(예외)",
                        value=str(st.session_state.get("ui_cf_any2_gpu_edit") or "powervr"),
                        key="ui_cf_any2_gpu_edit",
                        placeholder="예: powervr",
                        help="기본 조건을 못 맞춰도, GPU 문자열에 이 값이 포함되면 예외(any_of)로 후보에 포함합니다. (대소문자 무시)",
                    )

                # actions
                cbtn1, cbtn2, cbtn3 = st.columns(3)
                if cbtn1.button("YAML 검증", use_container_width=True):
                    try:
                        tmp_path = policy_path.parent / f"__tmp_validate__{policy_path.name}"
                        tmp_path.write_text(str(st.session_state.get("policy_edit_yaml_preview") or merged_yaml), encoding="utf-8")
                        _ = load_policy_v2(tmp_path)
                        tmp_path.unlink(missing_ok=True)
                        st.success("OK: YAML 파싱 및 v2 정책 로드 성공")
                    except Exception as e:
                        st.error(f"검증 실패: {e}")

                if cbtn2.button("원본 다시 불러오기", use_container_width=True):
                    st.session_state["policy_edit_yaml_text"] = raw_text
                    st.session_state["policy_edit_yaml_preview"] = raw_text
                    # re-prefill UI from original
                    try:
                        o = yaml.safe_load(raw_text) or {}
                        if not isinstance(o, dict):
                            o = {}
                    except Exception:
                        o = {}
                    _prefill_policy_edit_ui_from_yaml(obj=o, policy_stem=policy_path.stem)
                    st.rerun()

                if cbtn3.button("저장(덮어쓰기)", use_container_width=True, type="primary"):
                    try:
                        # 1) rename (파일명 변경) if needed
                        new_stem = _safe_policy_filename(str(st.session_state.get("ui_policy_name_edit") or "").strip())
                        if not new_stem:
                            raise RuntimeError("프로젝트 정책 이름(파일명)이 비어 있습니다.")
                        new_path = policy_path.parent / f"{new_stem}.yaml"
                        if new_path.exists() and new_path != policy_path:
                            raise RuntimeError(f"이미 존재하는 파일명입니다: {new_path.name}")

                        hist = PROJECT_ROOT / "config" / "history"
                        hist.mkdir(parents=True, exist_ok=True)
                        backup = hist / f"{policy_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                        backup.write_text(raw_text, encoding="utf-8")

                        if new_path != policy_path:
                            policy_path.rename(new_path)
                            policy_path = new_path
                            st.session_state["policy_editor_pick"] = policy_path.stem
                            st.session_state["_policy_edit_ui_last_pick"] = policy_path.stem
                        # normalize unsupported YAML formats (e.g., all_of) before validate/save
                        try:
                            preview_yaml = str(st.session_state.get("policy_edit_yaml_preview") or merged_yaml)
                            obj_save = yaml.safe_load(preview_yaml) or {}
                            if not isinstance(obj_save, dict):
                                raise RuntimeError("YAML 최상위는 dict 형식이어야 합니다.")
                        except Exception as e:
                            raise RuntimeError(f"YAML 파싱 실패: {e}")

                        obj_save2, warns = _normalize_policy_obj_to_supported_v2(obj_save)
                        if warns:
                            st.warning("\n".join([str(x) for x in warns if str(x).strip()][:5]))
                        final_yaml = yaml.safe_dump(obj_save2, allow_unicode=True, sort_keys=False)
                        st.session_state["policy_edit_yaml_preview"] = final_yaml
                        st.session_state["policy_edit_yaml_text"] = final_yaml

                        # validate before overwrite
                        tmp_path = policy_path.parent / f"__tmp_validate__{policy_path.name}"
                        tmp_path.write_text(final_yaml, encoding="utf-8")
                        _ = load_policy_v2(tmp_path)
                        tmp_path.unlink(missing_ok=True)
                        policy_path.write_text(final_yaml, encoding="utf-8")
                        _toast_once(f"저장 완료: {policy_path.name}", icon="✅")
                        # 저장 후 바로 추출 화면으로 이동(요청)
                        st.session_state["_policy_edit_just_saved"] = True
                        st.session_state["page_mode"] = "run"
                        # policy editor query params 제거
                        try:
                            _qp_set({})
                        except Exception:
                            pass
                        st.rerun()
                    except Exception as e:
                        st.error(f"저장 실패: {e}")

            with c2:
                st.subheader("정책 YAML (미리보기)")
                st.text_area("YAML", key="policy_edit_yaml_preview", height=840, disabled=True)
                with st.expander("고급: YAML 직접 편집(선택)", expanded=False):
                    st.caption("필요한 경우에만 직접 편집하세요. 일반적으로는 좌측 UI 입력값이 YAML에 반영됩니다.")
                    st.text_area("YAML(직접 편집)", key="policy_edit_yaml_text", height=420)

            return

    # screen == "create"
    st.subheader("정책 추가")
    st.caption("새 정책을 만들고 저장합니다. (기존 정책 파일은 변경되지 않습니다)")

    c1, c2 = st.columns(2)
    with c1:
            # NOTE: 현재는 프롬프트 템플릿을 UI에서 노출하지 않는다(기능은 유지).
            if False:
                with st.expander("프롬프트 템플릿(복사해서 사용)", expanded=False):
                    st.code("\n".join(prompt_template_lines), language="text")

            # UI 입력은 접힘/펼치기 없이 전체 노출
            st.caption(
                "방법 1) 위 입력값을 채운 뒤 ‘요구사항 문장 자동 생성’을 누르면 프롬프트가 생성되고, YAML 초안이 오른쪽에 자동으로 채워집니다. "
                "방법 2) 아래 ‘정책 프롬프트 확인/입력’에 직접 작성한 뒤 ‘YAML 초안 생성(OpenAI)’로 YAML만 생성할 수도 있습니다."
            )
            policy_name = st.text_input(
                "프로젝트 정책 이름 (필수)",
                value="",
                key="ui_policy_name_create",
                placeholder="예: CUSTOM (My Project)",
            )
            policy_desc = st.text_input("설명", value="", key="ui_policy_desc_create", placeholder="정책 설명")
            # 프로젝트 코드는 UI에서 직접 받지 않고, '프로젝트 정책 이름(필수)'을 우선 사용한다.
            # - 목적: 프롬프트/저장 YAML에서 project가 사용자가 입력한 정책 이름과 일치하도록 함
            # - fallback: 현재 선택된 정책(st.session_state["policy_editor_pick"]) 또는 NEWPROJ
            ui_project = str((policy_name or "").strip() or (st.session_state.get("policy_editor_pick") or "NEWPROJ")).strip()
            # 플랫폼은 사이드바에서 선택하는 값을 그대로 사용(중복 UI 제거)
            ui_platforms: list[str] = []
            if st.session_state.get("run_platform_android"):
                ui_platforms.append("android")
            if st.session_state.get("run_platform_ios"):
                ui_platforms.append("ios")
            m1, m2 = st.columns(2)
            manu_txt = m1.text_input(
                "제조사 필터(정확히 일치, 쉼표, 대소문자 무시)",
                value="",
                key="ui_cf_manufacturer_create",
                placeholder="예: SAMSUNG, APPLE",
                help="선택한 제조사의 디바이스만 후보에 포함됩니다. 비워두면 모든 제조사를 포함합니다.",
            )
            with m2:
                st.caption("아키텍처(정확히 일치, 선택)")
                a1, a2 = st.columns(2)
                with a1:
                    arch_64 = st.checkbox("64bit", value=True, key="ui_cf_arch_64_create")
                with a2:
                    arch_32 = st.checkbox("32bit", value=True, key="ui_cf_arch_32_create")
            gpu_contains = st.text_input(
                "GPU 포함(기본 필터)",
                value="",
                key="ui_cf_gpu_contains_create",
                placeholder="예: powervr / adreno",
                help="“이 GPU가 들어간 기기만 보고 싶다” → 이 조건을 만족하는 것만 후보로 남깁니다. (대소문자 무시)",
            )
            n1, n2, n3 = st.columns(3)

            # RAM 범위(선택): 체크했을 때만 min/max 표시 + 정수 입력
            ram_rng_on = n1.checkbox("RAM 범위", value=False, key="ui_cf_ram_rng_on_create", help="예: 2~3GB")
            ram_lo = 2
            ram_hi = 3
            if ram_rng_on:
                r1, r2 = n1.columns(2)
                ram_lo = int(
                    r1.number_input(
                        "최소(GB)",
                        min_value=0,
                        value=int(st.session_state.get("ui_cf_ram_lo_create", 2) or 2),
                        step=1,
                        format="%d",
                        key="ui_cf_ram_lo_create",
                    )
                )
                ram_hi = int(
                    r2.number_input(
                        "최대(GB)",
                        min_value=0,
                        value=int(st.session_state.get("ui_cf_ram_hi_create", 3) or 3),
                        step=1,
                        format="%d",
                        key="ui_cf_ram_hi_create",
                    )
                )
                n1.caption("RAM(GB) 최소~최대 범위만 후보로 통과합니다.")

                # OS 최소(선택): 체크했을 때만 입력 표시 (정수/버전문자열 모두 허용)
            os_min_on = n2.checkbox("OS 최소", value=False, key="ui_cf_os_min_on_create", help="예: Android 9+, iOS 15+")
            os_min_txt = ""
            if os_min_on:
                os_min_txt = n2.text_input(
                    "OS 버전(이상)",
                    value=str(st.session_state.get("ui_cf_os_min_text_create", "") or "9"),
                    key="ui_cf_os_min_text_create",
                    placeholder="예: 9 / 15 / 17.0.1",
                    help="예: 9 이상이면 Android 9+, iOS 15 이상이면 iOS 15+ 같은 의미",
                ).strip()
                n2.caption("입력한 버전 이상(>=)만 후보로 통과합니다.")

                # OR 대신: 특정 GPU 포함(선택)
            gpu_or_on = n3.checkbox(
                "GPU 포함(기본조건 만족 못해도 포함)",
                value=False,
                key="ui_cf_gpu_or_on_create",
                help="기본 조건은 유지하되, 해당 GPU도 포함해서 보고싶다",
            )
            any2_gpu = ""
            if gpu_or_on:
                any2_gpu = n3.text_input(
                    "GPU 포함 문자열(예외)",
                    value="",
                    key="ui_cf_any2_gpu_create",
                    placeholder="예: powervr",
                    help="기본 조건을 못 맞춰도, GPU 문자열에 이 값이 포함되면 예외(any_of)로 후보에 포함합니다. (대소문자 무시)",
                ).strip()

            if st.button("요구사항 문장 자동 생성", use_container_width=True, key="btn_fill_natural_create"):
                plats = ui_platforms if ui_platforms else ["android", "ios"]
                lines: list[str] = []
                if _safe_str(policy_name):
                    lines.append(f"프로젝트 정책 이름: {policy_name.strip()}")
                if _safe_str(policy_desc):
                    lines.append(f"설명: {policy_desc.strip()}")
                lines.append(f"프로젝트: {ui_project}")
                lines.append(f"플랫폼: {', '.join(plats)}")
                lines.append("후보 필터(AND):")
                manu = _unique_preserve_order(_split_csv_list(manu_txt))
                manu_norm = [x.upper() for x in manu]
                if manu:
                    lines.append(f"- 제조사(manufacturer) 정확히 일치(대소문자 무시): {', '.join(manu_norm)}")
                # 아키텍처: 둘 다 체크면 제한 없음(표시하지 않음). 하나만 체크면 해당 값으로 필터.
                arch_sel: list[str] = []
                if arch_64:
                    arch_sel.append("64bit")
                if arch_32:
                    arch_sel.append("32bit")
                if len(arch_sel) == 1:
                    lines.append(f"- 아키텍처(architecture) 정확히 일치: {arch_sel[0]}")
                if _safe_str(gpu_contains):
                    lines.append(f"- GPU 문자열(gpu) 포함: {gpu_contains.strip()}")
                if ram_rng_on:
                    lo, hi = float(ram_lo), float(ram_hi)
                    if lo > hi:
                        lo, hi = hi, lo
                    lines.append(f"- RAM(ram_gb) 범위: {lo:g}~{hi:g}")
                if os_min_on and os_min_txt:
                    lines.append(f"- OS(os_ver) 최소(>=): {os_min_txt}")
                if gpu_or_on and _safe_str(any2_gpu):
                    lines.append("후보 필터(OR/예외 any_of):")
                    lines.append(f"- GPU(gpu) contains '{any2_gpu.strip()}' (대소문자 무시)")
                # 1) 자연어 요구사항 박스 채우기
                st.session_state["policy_natural_text_create"] = "\n".join(lines)

                # 2) YAML 초안도 바로 생성해서 우측 YAML 박스에 채운다.
                api_key = os.getenv("OPENAI_API_KEY") or ""
                if not _safe_str(api_key):
                    # 3초 팝업(키 안내) — API 연결이 없으면 생성 중단
                    st.info("OpenAI API 키 연결이 필요합니다. 사이드바의 'API 설정'에서 키를 입력하세요. (3초 후 닫힘)")
                    time.sleep(3)
                    st.session_state["show_api_dialog"] = True
                    st.rerun()
                try:
                    model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
                    sys_txt = (
                        "너는 QA 디바이스 추천 정책(v2)을 YAML로 작성하는 도우미다.\n"
                        "반드시 YAML만 출력하고, 불필요한 설명은 하지 마라.\n"
                        "중요: candidate_filter에는 platform/include_values/include_contains/numeric_ranges/min_values/any_of만 사용해라.\n"
                        "RAM 규칙(중요):\n"
                        "- 'RAM 4GB만' => candidate_filter.numeric_ranges.ram_gb: [4,4]\n"
                        "- 'RAM 1~3GB' 또는 'RAM 1~3GB만' => candidate_filter.numeric_ranges.ram_gb: [1,3]\n"
                        "- 'RAM 3GB 이상' => candidate_filter.min_values.ram_gb: 3\n"
                        "금지: candidate_filter.all_of, field/op/operator, equals/between 같은 DSL 포맷."
                    )
                    usr_txt = (
                        "아래 요구사항을 v2 정책 YAML로 변환해줘.\n"
                        "규칙:\n"
                        "- (필수) diversity.within_rank.must_cover에는 반드시 cpu_family를 포함해라\n"
                        "- (필수) dedupe.within_rank.key는 반드시 [cpu, gpu, ram_gb] 로 넣어라\n"
                        "- (필수) dedupe.within_rank.max_per_product_name는 반드시 1 로 넣어라\n"
                        "- (필수) manufacturer_policy.within_rank.mode는 반드시 soft_dedupe 로 넣어라\n"
                        "- (필수) manufacturer_policy.within_rank.penalty_weight는 반드시 0.5 로 넣어라\n"
                        "- (OR/예외) 'A 또는 B' 조건이 필요하면 candidate_filter.any_of를 사용해라\n"
                        "- (금지) NOTE/note 관련 설정은 YAML에 절대 넣지 마라\n\n"
                        f"요구사항:\n{st.session_state.get('policy_natural_text_create','')}\n"
                    )
                    out = _openai_chat(api_key=api_key, model=model, user_text=usr_txt, system_text=sys_txt)
                    obj = yaml.safe_load(out) or {}
                    if not isinstance(obj, dict):
                        raise RuntimeError("OpenAI 응답 YAML 최상위가 dict가 아닙니다.")
                    # normalize/patch up to supported v2 schema
                    obj, warns = _normalize_policy_obj_to_supported_v2(obj)
                    # Reflect UI inputs into YAML (project/description) and correct common misses
                    ui_policy_name = str(st.session_state.get("ui_policy_name_create") or policy_name or "").strip()
                    ui_policy_desc = str(st.session_state.get("ui_policy_desc_create") or policy_desc or "").strip()

                    obj, warns2 = _heuristic_fill_missing_filters_from_text(
                        obj,
                        requirement_text=str(st.session_state.get("policy_natural_text_create", "")),
                        default_project=ui_policy_name or (str(ui_project).strip() or "NEWPROJ"),
                        default_platforms=(ui_platforms if ui_platforms else ["android", "ios"]),
                    )
                    # Heuristic may re-introduce root-level filters (e.g., numeric_ranges.ram_gb) that should be
                    # promoted into any_of[0] for A∪B semantics. Normalize once more after heuristic.
                    obj, warns3 = _normalize_policy_obj_to_supported_v2(obj)
                    # If policy name/description were provided in UI, enforce them into YAML
                    if ui_policy_name:
                        obj["project"] = ui_policy_name
                    if ui_policy_desc:
                        obj["description"] = ui_policy_desc

                    # Enforce create UI inputs into YAML (override LLM output)
                    try:
                        obj = _apply_policy_create_ui_overrides(base_obj=obj, ui_project=(ui_policy_name or ui_project), ui_platforms=plats)
                    except Exception:
                        pass
                    # IMPORTANT: override may re-introduce root-level filters while any_of exists.
                    # Normalize once more so any_of keeps A∪B semantics (base promoted into any_of[0]).
                    obj, warns4 = _normalize_policy_obj_to_supported_v2(obj)
                    all_warns = [*warns, *warns2, *warns3, *warns4]
                    if all_warns:
                        st.warning("\n".join([str(x) for x in all_warns if str(x).strip()][:6]))
                    st.session_state["_policy_create_yaml_source"] = "ui"
                    st.session_state["policy_create_yaml_text"] = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
                    st.rerun()
                except Exception:
                    st.info("YAML 초안 생성에 실패했습니다. API 설정(키/모델) 상태를 확인하세요. (3초 후 닫힘)")
                    time.sleep(3)
                    st.session_state["show_api_dialog"] = True
                    st.rerun()

            natural = st.text_area(
                "정책 프롬프트 확인/입력",
                key="policy_natural_text_create",
                placeholder=(
                    "\n".join([f"예시 {i+1}) {p}" for i, p in enumerate(_example_prompts_10())])
                    + "\n\nRAM 표현 예시) RAM 4GB만 / RAM 1~3GB만 / RAM 3GB 이상\n"
                ),
                height=_auto_textarea_height(
                    str(st.session_state.get("policy_natural_text_create", "")),
                    min_h=320,
                ),
            )

            # --- Debug/validation: run OpenAI for the 10 example prompts and show YAML results ---
            with st.expander("디버그: 예시 10개 OpenAI YAML 생성/검증(버튼 1번)", expanded=False):
                st.caption("예시 10개 문장을 OpenAI에 실제로 요청해 YAML을 생성하고, v2 로더로 검증합니다. 실패/누락은 빨간 로그로 표시합니다.")
                if st.button("예시 10개 실행(생성+검증)", use_container_width=True, key="btn_run_10_examples"):
                    api_key = os.getenv("OPENAI_API_KEY") or ""
                    if not _safe_str(api_key):
                        st.info("OpenAI API 키 연결이 필요합니다. 사이드바의 'API 설정'에서 키를 입력하세요. (3초 후 닫힘)")
                        time.sleep(3)
                        st.session_state["show_api_dialog"] = True
                        st.rerun()

                    model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
                    sys_txt = (
                        "너는 QA 디바이스 추천 정책(v2)을 YAML로 작성하는 도우미다.\n"
                        "반드시 YAML만 출력하고, 불필요한 설명은 하지 마라.\n"
                        "중요: candidate_filter에는 platform/include_values/include_contains/numeric_ranges/min_values/any_of만 사용해라.\n"
                        "RAM 규칙(중요):\n"
                        "- 'RAM 4GB만' => candidate_filter.numeric_ranges.ram_gb: [4,4]\n"
                        "- 'RAM 1~3GB' 또는 'RAM 1~3GB만' => candidate_filter.numeric_ranges.ram_gb: [1,3]\n"
                        "- 'RAM 3GB 이상' => candidate_filter.min_values.ram_gb: 3\n"
                        "금지: candidate_filter.all_of, field/op/operator, equals/between 같은 DSL 포맷."
                    )
                    prog = st.progress(0, text="예시 10개 실행 중…")
                    results: list[dict[str, Any]] = []
                    examples = _example_prompts_10()

                    for idx, prompt in enumerate(examples, start=1):
                        prog.progress(int((idx - 1) / len(examples) * 100), text=f"[{idx}/10] 생성/검증 중…")
                        row: dict[str, Any] = {"idx": idx, "prompt": prompt, "ok": False, "warns": [], "error": "", "yaml": ""}
                        try:
                            usr_txt = (
                                "아래 요구사항을 v2 정책 YAML로 변환해줘.\n"
                                "규칙:\n"
                                "- (필수) diversity.within_rank.must_cover에는 반드시 cpu_family를 포함해라\n"
                                "- (필수) dedupe.within_rank.key는 반드시 [cpu, gpu, ram_gb] 로 넣어라\n"
                                "- (필수) dedupe.within_rank.max_per_product_name는 반드시 1 로 넣어라\n"
                                "- (필수) manufacturer_policy.within_rank.mode는 반드시 soft_dedupe 로 넣어라\n"
                                "- (필수) manufacturer_policy.within_rank.penalty_weight는 반드시 0.5 로 넣어라\n"
                                "- (OR/예외) 'A 또는 B' 조건이 필요하면 candidate_filter.any_of를 사용해라\n"
                                "- (금지) NOTE/note 관련 설정은 YAML에 절대 넣지 마라\n\n"
                                f"요구사항:\n{prompt}\n"
                            )
                            out = _openai_chat(api_key=api_key, model=model, user_text=usr_txt, system_text=sys_txt)
                            obj = yaml.safe_load(out) or {}
                            if not isinstance(obj, dict):
                                raise RuntimeError("OpenAI 응답 YAML 최상위가 dict가 아닙니다.")
                            obj, w1 = _normalize_policy_obj_to_supported_v2(obj)
                            obj, w2 = _heuristic_fill_missing_filters_from_text(
                                obj,
                                requirement_text=prompt,
                                default_project=f"EXAMPLE_{idx:02d}",
                                default_platforms=["android", "ios"],
                            )
                            obj, w3 = _normalize_policy_obj_to_supported_v2(obj)
                            warns_all = [*w1, *w2, *w3]
                            row["warns"] = warns_all
                            final_yaml = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
                            row["yaml"] = final_yaml

                            # validate via loader (same as real save path)
                            tmp_path = CONFIG_V2_DIR / f"__tmp_validate__example_{idx:02d}.yaml"
                            tmp_path.write_text(final_yaml, encoding="utf-8")
                            _ = load_policy_v2(tmp_path)
                            tmp_path.unlink(missing_ok=True)
                            row["ok"] = True
                        except Exception as e:
                            row["error"] = str(e)
                            try:
                                tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
                            except Exception:
                                pass
                        results.append(row)

                    prog.progress(100, text="완료")
                    st.session_state["example10_results"] = results
                    st.rerun()

                res = st.session_state.get("example10_results") or []
                if isinstance(res, list) and res:
                    ok_cnt = sum(1 for r in res if isinstance(r, dict) and r.get("ok"))
                    st.write(f"결과: {ok_cnt}/10 OK")
                    for r in res:
                        if not isinstance(r, dict):
                            continue
                        idx = r.get("idx")
                        prompt = r.get("prompt")
                        ok = bool(r.get("ok"))
                        title = f"예시 {idx}) {'OK' if ok else 'FAIL'} — {prompt}"
                        with st.expander(title, expanded=not ok):
                            if ok:
                                st.success("검증 OK (load_policy_v2 통과)")
                            else:
                                st.error(f"검증 FAIL: {r.get('error')}")
                            warns = r.get("warns") or []
                            if warns:
                                st.warning("\n".join([str(x) for x in warns if str(x).strip()][:10]))
                            ytxt = str(r.get("yaml") or "").strip()
                            if ytxt:
                                st.code(ytxt)

            if st.button("YAML 초안 생성(OpenAI)", use_container_width=True):
                try:
                    api_key = os.getenv("OPENAI_API_KEY") or ""
                    model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
                    sys_txt = (
                        "너는 QA 디바이스 추천 정책(v2)을 YAML로 작성하는 도우미다.\n"
                        "반드시 YAML만 출력하고, 불필요한 설명은 하지 마라.\n"
                        "중요: candidate_filter에는 platform/include_values/include_contains/numeric_ranges/min_values/any_of만 사용해라.\n"
                        "RAM 규칙(중요):\n"
                        "- 'RAM 4GB만' => candidate_filter.numeric_ranges.ram_gb: [4,4]\n"
                        "- 'RAM 1~3GB' 또는 'RAM 1~3GB만' => candidate_filter.numeric_ranges.ram_gb: [1,3]\n"
                        "- 'RAM 3GB 이상' => candidate_filter.min_values.ram_gb: 3\n"
                        "금지: candidate_filter.all_of, field/op/operator, equals/between 같은 DSL 포맷."
                    )
                    usr_txt = (
                        "아래 요구사항을 v2 정책 YAML로 변환해줘.\n"
                        "규칙:\n"
                        "- (필수) diversity.within_rank.must_cover에는 반드시 cpu_family를 포함해라\n"
                        "- (필수) dedupe.within_rank.key는 반드시 [cpu, gpu, ram_gb] 로 넣어라\n"
                        "- (필수) dedupe.within_rank.max_per_product_name는 반드시 1 로 넣어라\n"
                        "- (필수) manufacturer_policy.within_rank.mode는 반드시 soft_dedupe 로 넣어라\n"
                        "- (필수) manufacturer_policy.within_rank.penalty_weight는 반드시 0.5 로 넣어라\n"
                        "- (OR/예외) 'A 또는 B' 조건이 필요하면 candidate_filter.any_of를 사용해라\n"
                        "- (금지) NOTE/note 관련 설정은 YAML에 절대 넣지 마라\n\n"
                        f"요구사항:\n{natural}\n"
                    )
                    out = _openai_chat(api_key=api_key, model=model, user_text=usr_txt, system_text=sys_txt)
                    obj = yaml.safe_load(out) or {}
                    if not isinstance(obj, dict):
                        raise RuntimeError("OpenAI 응답 YAML 최상위가 dict가 아닙니다.")
                    obj, warns = _normalize_policy_obj_to_supported_v2(obj)
                    ui_policy_name = str(st.session_state.get("ui_policy_name_create") or policy_name or "").strip()
                    ui_policy_desc = str(st.session_state.get("ui_policy_desc_create") or policy_desc or "").strip()

                    obj, warns2 = _heuristic_fill_missing_filters_from_text(
                        obj,
                        requirement_text=str(natural or ""),
                        default_project=ui_policy_name or (str(ui_project).strip() or "NEWPROJ"),
                        default_platforms=(ui_platforms if ui_platforms else ["android", "ios"]),
                    )
                    obj, warns3 = _normalize_policy_obj_to_supported_v2(obj)
                    if ui_policy_name:
                        obj["project"] = ui_policy_name
                    if ui_policy_desc:
                        obj["description"] = ui_policy_desc

                    # IMPORTANT: 프롬프트(직접 입력) 기반 생성은 UI 입력과 독립이어야 한다.
                    # - 여기서 UI로 candidate_filter를 덮어쓰지 않는다.
                    # - 저장 시에도 YAML을 그대로 보존할 수 있도록 source 플래그를 남긴다.
                    st.session_state["_policy_create_yaml_source"] = "prompt"
                    # IMPORTANT: override may re-introduce root-level filters while any_of exists.
                    # Normalize once more so any_of keeps A∪B semantics (base promoted into any_of[0]).
                    obj, warns4 = _normalize_policy_obj_to_supported_v2(obj)
                    all_warns = [*warns, *warns2, *warns3, *warns4]
                    if all_warns:
                        st.warning("\n".join([str(x) for x in all_warns if str(x).strip()][:6]))
                    st.session_state["policy_create_yaml_text"] = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with c2:
        st.subheader("새 정책 YAML")
        st.text_area("YAML", key="policy_create_yaml_text", height=840)
        st.markdown("#### 새 파일로 저장")
        default_name = f"{policy_path.stem}_new_{datetime.now().strftime('%m%d_%H%M')}"
        new_name = st.text_input("새 정책 파일명(.yaml 없이)", value=default_name, key="new_policy_name_create")
        new_stem = _safe_policy_filename(new_name)
        if st.button("새 파일로 저장", use_container_width=True, type="primary", key="btn_save_new_policy"):
            if not new_stem:
                st.error("파일명이 비어 있습니다.")
            else:
                out_path = CONFIG_V2_DIR / f"{new_stem}.yaml"
                if out_path.exists():
                    st.error(f"이미 존재하는 파일명입니다: {out_path.name}")
                else:
                    try:
                        obj = yaml.safe_load(st.session_state.get("policy_create_yaml_text", ""))
                        if not isinstance(obj, dict):
                            raise RuntimeError("YAML 최상위는 dict 형식이어야 합니다.")

                        src = str(st.session_state.get("_policy_create_yaml_source") or "ui").strip().lower()
                        if src != "prompt":
                            # Final: UI 값을 YAML에 강제 반영 (UI 기반 생성은 UI가 source of truth)
                            try:
                                pn2 = str(st.session_state.get("ui_policy_name_create") or policy_name or "").strip()
                                obj = _apply_policy_create_ui_overrides(
                                    base_obj=obj,
                                    ui_project=(pn2 or ui_project),
                                    ui_platforms=(ui_platforms if ui_platforms else ["android", "ios"]),
                                    strict=True,
                                )
                            except Exception:
                                pass

                        # A안(추천): UI에 설정된 값이 있는데 YAML에 빠져 있으면 자동 주입해서 저장
                        # - 필수: project (없으면 저장/로딩이 실패함)
                        #   UI의 '프로젝트 정책 이름'이 있으면 그것을 우선 사용한다.
                        if not str(obj.get("project") or "").strip():
                            if src == "prompt":
                                # 프롬프트 기반 저장은 YAML을 최대한 그대로 유지하되, 필수 key만 보정한다.
                                obj["project"] = new_stem or "NEWPROJ"
                            else:
                                pn = str(st.session_state.get("ui_policy_name_create") or policy_name or "").strip()
                                obj["project"] = pn or (str(ui_project).strip() or "NEWPROJ")

                        # - 선택: description (UI 입력을 우선 반영)
                        if src != "prompt":
                            ui_desc = str(st.session_state.get("ui_policy_desc_create") or policy_desc or "").strip()
                            if ui_desc:
                                obj["description"] = ui_desc

                        # - 선택: version
                        if not str(obj.get("version") or "").strip():
                            obj["version"] = "v2"

                        # - 선택: candidate_filter (platform/include_values/include_contains/numeric_ranges/min_values/any_of)
                        cf = obj.get("candidate_filter")
                        if not isinstance(cf, dict):
                            cf = {}
                            obj["candidate_filter"] = cf

                        # prompt 저장은 candidate_filter를 그대로 보존한다(필수 platform만 비어있으면 채움)
                        if src == "prompt":
                            plats = ui_platforms if ui_platforms else ["android", "ios"]
                            if not str(cf.get("platform") or "").strip():
                                cf["platform"] = ",".join(plats)
                            # normalize/validate only
                            obj, warns = _normalize_policy_obj_to_supported_v2(obj)
                            if warns:
                                st.warning("\n".join([str(x) for x in warns if str(x).strip()][:5]))
                            tmp_path = out_path.parent / f"__tmp_validate__{out_path.name}"
                            tmp_path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
                            _ = load_policy_v2(tmp_path)
                            tmp_path.unlink(missing_ok=True)
                            out_path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
                            _toast_once(f"저장 완료: {out_path.name}", icon="✅")
                            st.session_state["page_mode"] = "run"
                            st.session_state["policy_screen"] = "catalog"
                            st.rerun()

                        # platform
                        plats = ui_platforms if ui_platforms else ["android", "ios"]
                        if not str(cf.get("platform") or "").strip():
                            cf["platform"] = ",".join(plats)

                        # include_values.manufacturer (UI 제조사)
                        manu = _unique_preserve_order(_split_csv_list(str(st.session_state.get("ui_cf_manufacturer_create") or "")))
                        if manu:
                            iv = cf.get("include_values")
                            if not isinstance(iv, dict):
                                iv = {}
                                cf["include_values"] = iv
                            if not iv.get("manufacturer"):
                                iv["manufacturer"] = [x.upper() for x in manu]

                        # include_values.architecture (64bit/32bit 단일 선택일 때만)
                        arch_sel: list[str] = []
                        if bool(st.session_state.get("ui_cf_arch_64_create")):
                            arch_sel.append("64bit")
                        if bool(st.session_state.get("ui_cf_arch_32_create")):
                            arch_sel.append("32bit")
                        if len(arch_sel) == 1:
                            iv = cf.get("include_values")
                            if not isinstance(iv, dict):
                                iv = {}
                                cf["include_values"] = iv
                            # always override: UI is the source of truth
                            iv["architecture"] = [arch_sel[0]]
                        else:
                            # both checked(or none) => no restriction
                            iv = cf.get("include_values")
                            if isinstance(iv, dict):
                                iv.pop("architecture", None)

                        # remove invalid include_contains.architecture if present (LLM mistake)
                        ic = cf.get("include_contains")
                        if isinstance(ic, dict) and ("architecture" in ic):
                            ic.pop("architecture", None)
                            if not ic:
                                cf.pop("include_contains", None)

                        # include_contains.gpu (기본 필터)
                        gpu_basic = str(st.session_state.get("ui_cf_gpu_contains_create") or "").strip()
                        if gpu_basic:
                            ic = cf.get("include_contains")
                            if not isinstance(ic, dict):
                                ic = {}
                                cf["include_contains"] = ic
                            if not ic.get("gpu"):
                                ic["gpu"] = [gpu_basic]

                        # numeric_ranges.ram_gb (RAM 범위)
                        if bool(st.session_state.get("ui_cf_ram_rng_on_create")):
                            lo = float(st.session_state.get("ui_cf_ram_lo_create", 2) or 2)
                            hi = float(st.session_state.get("ui_cf_ram_hi_create", 3) or 3)
                            if lo > hi:
                                lo, hi = hi, lo
                            nr = cf.get("numeric_ranges")
                            if not isinstance(nr, dict):
                                nr = {}
                                cf["numeric_ranges"] = nr
                            if not nr.get("ram_gb"):
                                nr["ram_gb"] = [lo, hi]

                        # min_values.os_ver (OS 최소) - 숫자로 변환 가능할 때만 주입
                        if bool(st.session_state.get("ui_cf_os_min_on_create")):
                            os_min_txt = str(st.session_state.get("ui_cf_os_min_text_create") or "").strip()
                            if os_min_txt:
                                try:
                                    os_min_val = float(os_min_txt)
                                except Exception:
                                    os_min_val = None
                                if os_min_val is not None:
                                    mv = cf.get("min_values")
                                    if not isinstance(mv, dict):
                                        mv = {}
                                        cf["min_values"] = mv
                                    if "os_ver" not in mv:
                                        mv["os_ver"] = os_min_val

                        # any_of (예외 GPU)
                        if bool(st.session_state.get("ui_cf_gpu_or_on_create")):
                            any2_gpu = str(st.session_state.get("ui_cf_any2_gpu_create") or "").strip()
                            if any2_gpu:
                                if not cf.get("any_of"):
                                    cf["any_of"] = [{"include_contains": {"gpu": [any2_gpu]}}]

                        # normalize unsupported YAML formats (e.g., all_of) before validate/save
                        obj, warns = _normalize_policy_obj_to_supported_v2(obj)
                        if warns:
                            st.warning("\n".join([str(x) for x in warns if str(x).strip()][:5]))

                        tmp_path = out_path.parent / f"__tmp_validate__{out_path.name}"
                        tmp_path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
                        _ = load_policy_v2(tmp_path)
                        tmp_path.unlink(missing_ok=True)
                        out_path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")
                        _toast_once(f"저장 완료: {out_path.name}", icon="✅")
                        # 저장 직후 바로 추출 화면으로 이동
                        st.session_state["page_mode"] = "run"
                        st.session_state["policy_screen"] = "catalog"
                        st.rerun()
                    except Exception as e:
                        st.error(f"저장 실패: {e}")

    return


def _run() -> None:
    st.set_page_config(page_title="devices_auto - Policy v2 POC", layout="wide")
    # --- Product-like base styling (CSS) ---
    st.markdown(
        """
<style>
/* App background */
div[data-testid="stAppViewContainer"]{
  background: #f5f7fb;
}
/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] *{
  color: #e5e7eb !important;
}
/* Sidebar widgets: make inputs readable on dark background */
section[data-testid="stSidebar"] div[data-baseweb="input"] input{
  background: rgba(255,255,255,0.06) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"]{
  background: rgba(255,255,255,0.06) !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="base-input"]{
  background: rgba(255,255,255,0.06) !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] textarea{
  background: rgba(255,255,255,0.06) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span{
  color: #e5e7eb !important;
}
section[data-testid="stSidebar"] .stNumberInput button{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: #e5e7eb !important;
}
section[data-testid="stSidebar"] .stFileUploader{
  background: rgba(255,255,255,0.04) !important;
  border-radius: 14px !important;
  border: 1px dashed rgba(255,255,255,0.18) !important;
  padding: 10px 10px 2px 10px !important;
}
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"]{
  color: #e5e7eb !important;
}
section[data-testid="stSidebar"] .stTabs [aria-selected="true"]{
  border-bottom: 2px solid rgba(99,102,241,0.95) !important;
}
/* Sidebar buttons (e.g., 정책확인 버튼) */
section[data-testid="stSidebar"] .stButton > button{
  background: rgba(255,255,255,0.06) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover{
  background: rgba(255,255,255,0.10) !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled{
  background: rgba(255,255,255,0.04) !important;
  color: rgba(229,231,235,0.45) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}
/* Sidebar placeholders / hints */
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder{
  color: rgba(229,231,235,0.55) !important;
}
/* File uploader internals (dropzone is rendered with its own box) */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,0.04) !important;
  border: 1px dashed rgba(255,255,255,0.18) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{
  color: rgba(229,231,235,0.75) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{
  background: rgba(255,255,255,0.06) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover{
  background: rgba(255,255,255,0.10) !important;
}

/* Select popup menu (BaseWeb portal): force dark menu even when rendered outside sidebar */
div[data-baseweb="popover"]{
  background: #0b1220 !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  box-shadow: 0 12px 32px rgba(0,0,0,0.35) !important;
}
div[data-baseweb="popover"] > div{
  background: #0b1220 !important;
}
div[data-baseweb="popover"] *{
  color: #e5e7eb !important;
}
/* Some BaseWeb menus paint a white background on inner wrappers.
   Force the menu surface dark and keep option children transparent so the dark surface shows through. */
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [role="menu"],
div[data-baseweb="popover"] [role="listbox"]{
  background: #0b1220 !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li *,
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] *{
  background-color: transparent !important;
}
div[data-baseweb="popover"] input{
  background: rgba(255,255,255,0.06) !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
div[data-baseweb="popover"] input::placeholder{
  color: rgba(229,231,235,0.55) !important;
}
ul[data-baseweb="menu"]{
  background: #0b1220 !important;
}
div[data-baseweb="popover"] [role="listbox"]{
  background: #0b1220 !important;
}
/* DARK_DROPDOWN_V3
   Some Streamlit/BaseWeb versions render the listbox/menu outside the popover wrapper.
   Make ALL dropdown menus dark globally so sidebar text color doesn't clash. */
div[role="listbox"], ul[role="listbox"], ul[data-baseweb="menu"], div[data-baseweb="menu"], div[data-baseweb="layer"]{
  background: #0b1220 !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  box-shadow: 0 12px 32px rgba(0,0,0,0.35) !important;
}
div[role="listbox"] *, ul[role="listbox"] *, ul[data-baseweb="menu"] *, div[data-baseweb="menu"] *{
  color: #e5e7eb !important;
}
/* Options can be <li role="option"> or <div role="option"> depending on BaseWeb version */
li[role="option"], div[role="option"]{
  background: transparent !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"]:hover,
div[role="option"][aria-selected="true"]{
  background: rgba(99,102,241,0.18) !important;
}
/* Buttons */
.stButton > button{
  border-radius: 12px;
  padding: 0.55rem 0.9rem;
  border: 1px solid rgba(0,0,0,0.10);
}
/* Inputs */
div[data-baseweb="input"] input, textarea{
  border-radius: 10px !important;
}
/* Headers */
h1, h2, h3{
  letter-spacing: -0.02em;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    # page mode (run vs policy)
    st.session_state.setdefault("page_mode", "run")
    if st.session_state.get("page_mode") != "policy":
        st.title("QA 디바이스 자동 추출 시스템 v2")
        st.caption("Policy 기반 선택 + Rank 목표 입력 → 중복 제거/다양성/타이브레이커 → 결과/엑셀 다운로드")
    # (요청) build/python 표시 제거

    _ensure_repo_root_on_path()

    from app.policy_v2 import CandidateFilter, load_policy_v2
    from app.policy_v2_engine import PolicyV2RunInputs, run_policy_v2
    from app.testbed_normalizer import load_testbed, normalize_testbed
    from app.env import load_default_env

    # load local.env/.env best-effort so keys are available
    load_default_env(PROJECT_ROOT, override=False)

    # --- API 설정 확인(버튼 → 팝업) ---
    st.sidebar.header("API 설정 확인")
    st.session_state.setdefault("show_api_dialog", False)
    local_env_path = PROJECT_ROOT / "local.env"
    existing = _read_env_kv(local_env_path)

    # Auto healthcheck on app start (once per session unless key changes)
    # - 목적: local.env/환경변수에 키가 이미 있는 경우 '미적용' 대신 정상/비정상 상태를 자동 표시
    st.session_state.setdefault("api_status", "미적용")
    cur_key = _safe_str(os.getenv("OPENAI_API_KEY") or existing.get("OPENAI_API_KEY", ""))
    last_key = str(st.session_state.get("_api_status_last_key") or "")
    last_ts = float(st.session_state.get("_api_status_last_ts") or 0.0)
    now_ts = float(time.time())
    # throttle: same key면 60초 내 재검사하지 않음
    should_check = bool(cur_key) and ((cur_key != last_key) or (now_ts - last_ts > 60.0))
    if should_check:
        try:
            ok, msg = _openai_healthcheck(cur_key)
            st.session_state["api_status"] = "정상" if ok else f"비정상({msg})"
        except Exception as e:
            st.session_state["api_status"] = f"비정상({e})"
        st.session_state["_api_status_last_key"] = cur_key
        st.session_state["_api_status_last_ts"] = now_ts

    if st.sidebar.button("API 설정", use_container_width=True, key="btn_api_settings_sidebar"):
        st.session_state["show_api_dialog"] = True
        st.rerun()

    def _render_api_settings(*, allow_save_local: bool) -> None:
        st.caption("세션 적용: 현재 실행 중인 Streamlit 프로세스에만 반영됩니다.")
        if allow_save_local:
            st.caption("저장을 켜면 local.env에 기록됩니다.")

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

        st.session_state.setdefault("api_status", "미적용")
        st.info(f"상태: {st.session_state.get('api_status')}")

        save_local = False
        if allow_save_local:
            save_local = st.checkbox("local.env에 저장", value=True)

        c_ap1, c_ap2 = st.columns([1, 1])
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

            if allow_save_local and save_local:
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

            ok, msg = _openai_healthcheck(openai_key)
            st.session_state["api_status"] = "정상" if ok else f"비정상({msg})"
            if ok:
                st.success(
                    "세션 정상 연결 (3초 후 닫힘)"
                    + (" + local.env 저장" if (allow_save_local and save_local) else "")
                )
                time.sleep(3)
                st.session_state["show_api_dialog"] = False
                st.rerun()
            else:
                st.error(f"연결 실패: {msg}")

        if allow_save_local and c_ap2.button("local.env 다시 읽기", use_container_width=True):
            st.session_state["show_api_dialog"] = False
            st.rerun()

    if st.session_state.get("show_api_dialog"):
        if hasattr(st, "dialog"):
            @st.dialog("API 설정")
            def _api_dialog() -> None:
                _render_api_settings(allow_save_local=True)

            _api_dialog()
        else:
            # Fallback: older Streamlit without dialog support
            with st.sidebar.expander("API 설정", expanded=True):
                _render_api_settings(allow_save_local=True)

    st.sidebar.divider()

    # (요청) 사이드바 '입력' 문구 제거
    # ignore temporary validation artifacts (e.g., __tmp_validate__*.yaml)
    policy_files = sorted([p for p in CONFIG_V2_DIR.glob("*.yaml") if not str(p.name).startswith("__tmp_validate__")])
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

    policy_pick = st.sidebar.selectbox(
        "프로젝트 정책",
        options=policy_names,
        index=policy_names.index(st.session_state["policy_pick"]) if st.session_state["policy_pick"] in policy_names else 0,
        key="policy_pick",
    )
    policy_path = next(p for p in policy_files if p.stem == policy_pick)
    policy = load_policy_v2(policy_path)

    # 정책 센터 바로가기 버튼(설정/편집/생성)
    if st.sidebar.button("정책 설정/편집/생성", use_container_width=True, key="btn_open_policy_catalog"):
        # 항상 목록부터 시작(이전 세션의 edit/view/create 상태, 또는 query param이 남아도 강제 리셋)
        # - query param 복원 로직이 policy_screen=create 등을 다시 덮어쓸 수 있으므로,
        #   정책 화면 진입 버튼에서 URL을 catalog로 먼저 고정한다.
        def _set_policy_query_params(screen: str, policy_stem: str) -> None:
            try:
                qp = getattr(st, "query_params", None)
                if qp is not None:
                    try:
                        qp.clear()
                    except Exception:
                        pass
                    qp.update({"policy_screen": str(screen), "policy": str(policy_stem)})
                    return
            except Exception:
                pass
            try:
                fn = getattr(st, "experimental_set_query_params", None)
                if callable(fn):
                    fn(policy_screen=str(screen), policy=str(policy_stem))
            except Exception:
                pass

        st.session_state["show_api_dialog"] = False
        st.session_state["policy_screen"] = "catalog"
        _set_policy_query_params("catalog", str(st.session_state.get("policy_pick") or ""))
        st.session_state["page_mode"] = "policy"
        st.rerun()

    version = st.sidebar.text_input("릴리즈 버전", value=st.session_state.get("version", "4.3.0"), key="version")

    # upload first so we can build dropdown options (target country, etc.)
    st.sidebar.divider()
    upload = st.sidebar.file_uploader("마스터 디바이스 목록 업로드(Excel/CSV)", type=["xlsx", "xls", "csv"])

    # --- 실행 플랫폼 선택(UI) ---
    policy_platforms = _parse_platforms(getattr(policy.candidate_filter, "platform", "android"))
    if st.session_state.get("_run_last_policy_pick") != policy_pick:
        st.session_state["_run_last_policy_pick"] = policy_pick
        st.session_state["run_platform_android"] = "android" in policy_platforms
        st.session_state["run_platform_ios"] = "ios" in policy_platforms

    st.sidebar.subheader("플랫폼 선택")
    pcol1, pcol2 = st.sidebar.columns(2)
    with pcol1:
        st.checkbox("Android (AOS)", key="run_platform_android")
    with pcol2:
        st.checkbox("iOS", key="run_platform_ios")
    selected_platforms: list[str] = []
    if st.session_state.get("run_platform_android"):
        selected_platforms.append("android")
    if st.session_state.get("run_platform_ios"):
        selected_platforms.append("ios")
    if not selected_platforms:
        selected_platforms = policy_platforms or ["android"]

    platforms = selected_platforms

    def _rank_inputs(prefix: str) -> dict[str, int]:
        # (요청) 문구 정리: 플랫폼별 합계는 아래 caption으로 표시
        # 섹션 타이틀은 호출부에서 출력(탭 제거용)
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
        # migrate old android keys once (best-effort)
        if platforms[0] == "android":
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
            rank_targets_android = _rank_inputs("android")
            rank_targets_ios = {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0, "-": 0}
        else:
            rank_targets_android = {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0, "-": 0}
            rank_targets_ios = _rank_inputs("ios")
    else:
        st.sidebar.subheader("Rank별 목표 수량(AOS)")
        rank_targets_android = _rank_inputs("android")
        st.sidebar.subheader("Rank별 목표 수량(IOS)")
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
        st.sidebar.selectbox(
            "제외할 타겟 국가(중복 선택)",
            options=["디바이스 파일 업로드 후 자동 추출"],
            index=0,
            disabled=True,
        )
        exclude_targets = []
    else:
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
        st.sidebar.selectbox(
            "제외할 NOTE(중복 선택)",
            options=["디바이스 파일 업로드 후 자동 추출"],
            index=0,
            disabled=True,
        )
        exclude_notes = []
    else:
        exclude_notes = st.sidebar.multiselect(
            "제외할 NOTE(중복 선택)",
            options=note_opts,
            default=[],
        )

    required_nos = st.sidebar.text_input(
        "필수 디바이스 No(콤마, 선택)",
        value=st.session_state.get("required_nos", ""),
        key="required_nos",
        placeholder="선택 사항 - 지정 시 해당 디바이스 강제 포함",
    )
    prev_excl_nos = st.sidebar.text_input(
        "직전 버전 제외 No(콤마, 선택)",
        value=st.session_state.get("prev_excl_nos", ""),
        key="prev_excl_nos",
        placeholder="해당 No의 제품명/모델명을 이번 버전에서 제외",
        help="해당 No들의 제품명/모델명은 이번 버전에서 제외",
    )

    st.sidebar.divider()
    run_btn = st.sidebar.button("자동 추출 실행", type="primary", disabled=(upload is None))

    # 정책 관리 화면에서는, 사이드바는 동일하게 유지하고 본문만 정책 관리로 전환
    if st.session_state.get("page_mode") == "policy":
        # NOTE: policy 화면의 상단바(메인화면 이동/새 정책)는 _policy_editor_page()에서 렌더링한다.
        _policy_editor_page(policy_files=policy_files, selected_policy=policy_path)
        return

    st.subheader("결과")
    result_box = st.empty()
    download_box = st.empty()

    if not run_btn:
        with result_box.container():
            st.info(
                "사용 방법: 좌측 사이드바에서 프로젝트 정책과 Rank 목표를 설정한 후, "
                "테스트베드 파일을 업로드하고 '자동 추출 실행'을 클릭하세요."
            )
            st.markdown(
                """
<div style="background:#ffffff;border:1px solid rgba(0,0,0,0.06);border-radius:16px;padding:28px;text-align:center;">
  <div style="font-size:20px;font-weight:700;margin-bottom:6px;color:#111827;">결과가 여기에 표시됩니다</div>
  <div style="color:#6b7280;line-height:1.6;">
    좌측 사이드바에서 설정을 완료하고<br/>
    테스트베드 파일을 업로드한 후<br/>
    <b>"자동 추출 실행"</b> 버튼을 클릭하세요
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
<div style="margin-top:16px;background:linear-gradient(180deg, rgba(255,200,60,0.28) 0%, rgba(255,200,60,0.18) 100%);
border:1px solid rgba(255,180,0,0.25);border-radius:16px;padding:18px;">
  <div style="font-weight:800;color:#111827;margin-bottom:8px;">📌 5단계 선택 알고리즘</div>
  <div style="color:#111827;line-height:1.7;">
    <div>Stage 0: 필수 디바이스 강제 선택 (required_nos)</div>
    <div>Stage 1: 후보 필터링 (플랫폼, availability, 제외 조건)</div>
    <div>Stage 2: 프로파일 대표성 확보 (중복 제거)</div>
    <div>Stage 3: 다양성 확보 (must_cover_axes)</div>
    <div>Stage 4: 제조사 분산 (manufacturer soft_dedupe)</div>
    <div>Stage 5: 부족분 충족 (extras fill)</div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
        return

    total_n_android = int(sum(rank_targets_android.values())) if isinstance(rank_targets_android, dict) else 0
    total_n_ios = int(sum(rank_targets_ios.values())) if isinstance(rank_targets_ios, dict) else 0
    total_n_all = total_n_android + total_n_ios if len(platforms) == 2 else (total_n_android if platforms[0] == "android" else total_n_ios)
    st.sidebar.caption(f"Rank 목표 합계: Android={total_n_android} / iOS={total_n_ios} (현재 실행 합계={total_n_all})")
    if total_n_all <= 0:
        st.sidebar.error("Rank 목표 수량 합계가 0입니다.")
        return

    with st.spinner("업로드 파일 로딩/정규화 중..."):
        up_path = _save_upload(upload, prefix=f"testbed_{policy.project}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        raw_df = load_testbed(up_path)
        norm_df = normalize_testbed(raw_df)

    # (요청) 디버그 UI는 다운로드 섹션 아래로 이동 (결과 표 근처에 상시 노출하지 않음)

    inputs = PolicyV2RunInputs(
        project=policy.project,
        version=version,
        rank_targets=(rank_targets_android if (len(platforms) == 1 and platforms[0] == "android") else rank_targets_ios),
        required_nos=_split_csv_list(required_nos),
        exclude_prev_version_nos=_split_csv_list(prev_excl_nos),
        exclude_target_countries=list(exclude_targets or []),
        exclude_notes=list(exclude_notes or []),
    )

    def _candidate_filter_for_platform(cf: CandidateFilter, platform: str) -> CandidateFilter:
        """
        플랫폼별로 정책을 분기 실행할 때, 한쪽 플랫폼에만 의미 있는 필터가 다른 쪽을 0건으로 만들지 않도록 보정.

        - iOS는 사실상 단일 제조사(APPLE)이고, testbed에 architecture가 비어 있는 경우가 많다.
          따라서 iOS 실행 시 manufacturer/architecture 관련 조건은 보수적으로 무시(또는 APPLE 포함일 때만 유지).
        """
        from dataclasses import replace

        # copy dicts (CandidateFilter는 frozen dataclass)
        iv = dict(getattr(cf, "include_values", {}) or {})
        ic = dict(getattr(cf, "include_contains", {}) or {})
        mv = dict(getattr(cf, "min_values", {}) or {})
        nr = dict(getattr(cf, "numeric_ranges", {}) or {})

        # recurse any_of branches
        ao = tuple(_candidate_filter_for_platform(x, platform) for x in (getattr(cf, "any_of", None) or ()))

        if str(platform).strip().lower() == "ios":
            # manufacturer filter:
            # - keep only if APPLE is explicitly included
            # - otherwise drop it to avoid "KRJP(SAMSUNG)" blocking iOS completely
            manu_vals = iv.get("manufacturer", ())
            manu_set = {str(x).strip().upper() for x in (manu_vals or ()) if str(x).strip()}
            if manu_set and ("APPLE" not in manu_set):
                iv.pop("manufacturer", None)

            # architecture filter: iOS rows may have empty architecture; drop for iOS runs
            if "architecture" in iv:
                iv.pop("architecture", None)

            # common arch-bits keys sometimes appear in min_values/include_contains; drop for iOS runs
            for k in ("cpu_arch_bits", "os_arch_bits", "cpu_arch", "os_arch", "arch", "architecture"):
                mv.pop(k, None)
                ic.pop(k, None)

        return replace(
            cf,
            platform=platform,
            include_values=iv,
            include_contains=ic,
            min_values=mv,
            numeric_ranges=nr,
            any_of=ao,
        )

    with st.spinner("정책 기반 자동 추출 중..."):
        # 플랫폼이 android,ios 모두면 각각 따로 추출
        if len(platforms) == 2:
            from dataclasses import replace

            cf0 = policy.candidate_filter
            p_android = replace(policy, candidate_filter=_candidate_filter_for_platform(cf0, "android"))
            p_ios = replace(policy, candidate_filter=_candidate_filter_for_platform(cf0, "ios"))

            out_android = run_policy_v2(master_df=norm_df, policy=p_android, inputs=replace(inputs, rank_targets=rank_targets_android))
            out_ios = run_policy_v2(master_df=norm_df, policy=p_ios, inputs=replace(inputs, rank_targets=rank_targets_ios))
        else:
            from dataclasses import replace

            cf0 = policy.candidate_filter
            p_one = replace(policy, candidate_filter=_candidate_filter_for_platform(cf0, platforms[0]))
            out_android = run_policy_v2(master_df=norm_df, policy=p_one, inputs=inputs)
            out_ios = None

    # 예외 GPU 매칭 전체 목록(선정과 별개) 계산
    try:
        if out_ios is None:
            exc_df_a, exc_toks_a = _exception_gpu_matches_df(master_df=norm_df, policy=p_one)
            exc_df_i, exc_toks_i = (pd.DataFrame(), [])
        else:
            exc_df_a, exc_toks_a = _exception_gpu_matches_df(master_df=norm_df, policy=p_android)
            exc_df_i, exc_toks_i = _exception_gpu_matches_df(master_df=norm_df, policy=p_ios)
    except Exception:
        exc_df_a, exc_toks_a = (pd.DataFrame(), [])
        exc_df_i, exc_toks_i = (pd.DataFrame(), [])

    with result_box.container():
        if out_ios is None:
            st.subheader("Rank별 최종 선택(미리보기)")
            sel_with_why = _attach_why(out_android.selected, out_android.decision_log)
            cleaned = _dedupe_cols(_clean_result_df(sel_with_why))

            # UX: pin required devices to top (디버그 표시는 다운로드 섹션 아래로 이동)
            try:
                req_list_ui = _split_csv_list(st.session_state.get("required_nos", "") or "")
                req_set = {str(x).strip().upper() for x in req_list_ui if str(x).strip()}
                if req_set and "No" in cleaned.columns:
                    s_no = cleaned["No"].fillna("").astype(str).str.strip().str.upper()
                    m_req = s_no.isin(req_set)
                    req_rows_out = cleaned[m_req].copy()
                    if not req_rows_out.empty:
                        cleaned = pd.concat([req_rows_out, cleaned[~m_req]], ignore_index=True, sort=False)
            except Exception:
                pass
            st.dataframe(cleaned.head(600), use_container_width=True)

            # 예외 GPU 전체 목록(선정과 별개)
            if isinstance(exc_df_a, pd.DataFrame) and not exc_df_a.empty:
                st.subheader("예외 GPU 매칭 전체(선정과 별개)")
                st.caption(
                    f"예외 GPU: {', '.join(exc_toks_a) if exc_toks_a else '-'} / 총 {len(exc_df_a)}대. "
                    "이 목록은 dedupe/랭크 목표와 무관한 ‘매칭 전체’입니다."
                )
                exc_view = _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(exc_df_a), platforms[0] if platforms else "android"))
                # 예외 GPU 매칭 전체는 '선정' 결과가 아니므로 선정 사유 컬럼은 혼동을 유발할 수 있어 제거한다.
                exc_view = exc_view.drop(columns=["선정 사유"], errors="ignore")
                st.dataframe(exc_view.head(600), use_container_width=True)
        else:
            t1, t2 = st.tabs(["Android(AOS) 결과", "iOS 결과"])
            with t1:
                st.subheader("Rank별 최종 선택(미리보기)")
                sel_with_why = _attach_why(out_android.selected, out_android.decision_log)
                cleaned = _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(sel_with_why), "android"))
                # UX: pin required devices to top (디버그 표시는 다운로드 섹션 아래로 이동)
                try:
                    req_list_ui = _split_csv_list(st.session_state.get("required_nos", "") or "")
                    req_set = {str(x).strip().upper() for x in req_list_ui if str(x).strip()}
                    if req_set and "No" in cleaned.columns:
                        s_no = cleaned["No"].fillna("").astype(str).str.strip().str.upper()
                        m_req = s_no.isin(req_set)
                        req_rows_out = cleaned[m_req].copy()
                        if not req_rows_out.empty:
                            cleaned = pd.concat([req_rows_out, cleaned[~m_req]], ignore_index=True, sort=False)
                except Exception:
                    pass
                st.dataframe(cleaned.head(600), use_container_width=True)

                if isinstance(exc_df_a, pd.DataFrame) and not exc_df_a.empty:
                    st.subheader("예외 GPU 매칭 전체(선정과 별개)")
                    st.caption(
                        f"예외 GPU: {', '.join(exc_toks_a) if exc_toks_a else '-'} / 총 {len(exc_df_a)}대. "
                        "이 목록은 dedupe/랭크 목표와 무관한 ‘매칭 전체’입니다."
                    )
                    exc_view = _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(exc_df_a), "android"))
                    # 예외 GPU 매칭 전체는 '선정' 결과가 아니므로 선정 사유 컬럼은 혼동을 유발할 수 있어 제거한다.
                    exc_view = exc_view.drop(columns=["선정 사유"], errors="ignore")
                    st.dataframe(exc_view.head(600), use_container_width=True)
            with t2:
                st.subheader("Rank별 최종 선택(미리보기)")
                sel_with_why = _attach_why(out_ios.selected, out_ios.decision_log)
                cleaned = _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(sel_with_why), "ios"))
                # UX: pin required devices to top (디버그 표시는 다운로드 섹션 아래로 이동)
                try:
                    req_list_ui = _split_csv_list(st.session_state.get("required_nos", "") or "")
                    req_set = {str(x).strip().upper() for x in req_list_ui if str(x).strip()}
                    if req_set and "No" in cleaned.columns:
                        s_no = cleaned["No"].fillna("").astype(str).str.strip().str.upper()
                        m_req = s_no.isin(req_set)
                        req_rows_out = cleaned[m_req].copy()
                        if not req_rows_out.empty:
                            cleaned = pd.concat([req_rows_out, cleaned[~m_req]], ignore_index=True, sort=False)
                except Exception:
                    pass
                st.dataframe(cleaned.head(600), use_container_width=True)

                if isinstance(exc_df_i, pd.DataFrame) and not exc_df_i.empty:
                    st.subheader("예외 GPU 매칭 전체(선정과 별개)")
                    st.caption(
                        f"예외 GPU: {', '.join(exc_toks_i) if exc_toks_i else '-'} / 총 {len(exc_df_i)}대. "
                        "이 목록은 dedupe/랭크 목표와 무관한 ‘매칭 전체’입니다."
                    )
                    exc_view = _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(exc_df_i), "ios"))
                    # 예외 GPU 매칭 전체는 '선정' 결과가 아니므로 선정 사유 컬럼은 혼동을 유발할 수 있어 제거한다.
                    exc_view = exc_view.drop(columns=["선정 사유"], errors="ignore")
                    st.dataframe(exc_view.head(600), use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if out_ios is None:
            _dedupe_cols(out_android.rule_summary).to_excel(writer, sheet_name="Policy_Summary", index=False)
            for rk in ["A+", "A", "B", "C", "D", "-"]:
                df_rk = out_android.selected_by_rank.get(rk, pd.DataFrame())
                name = f"Selected_{rk}".replace("+", "PLUS").replace("-", "DASH")
                df_rk2 = _attach_why(df_rk, out_android.decision_log)
                _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(df_rk2), platforms[0] if platforms else "android")).to_excel(
                    writer, sheet_name=name[:31], index=False
                )
            all2 = _attach_why(out_android.selected, out_android.decision_log)
            _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(all2), platforms[0] if platforms else "android")).to_excel(
                writer, sheet_name="Selected_All", index=False
            )
            # 예외 GPU 매칭 전체(선정과 별개)
            if isinstance(exc_df_a, pd.DataFrame) and not exc_df_a.empty:
                exc_sheet = _dedupe_cols(
                    _postprocess_output_for_platform(_clean_result_df(exc_df_a), platforms[0] if platforms else "android")
                )
                exc_sheet.to_excel(writer, sheet_name="GPU_Exception_All", index=False)
        else:
            _dedupe_cols(out_android.rule_summary).to_excel(writer, sheet_name="Policy_Summary", index=False)
            # (요청) Shortage_AOS / Shortage_IOS 시트 생성 제거
            # (요청) AOS_DASH(-), IOS_D, IOS_DASH(-) 시트 생성 제거
            for rk in ["A+", "A", "B", "C", "D"]:
                df_rk = out_android.selected_by_rank.get(rk, pd.DataFrame())
                name = f"AOS_{rk}".replace("+", "PLUS").replace("-", "DASH")
                df_rk2 = _attach_why(df_rk, out_android.decision_log)
                _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(df_rk2), "android")).to_excel(writer, sheet_name=name[:31], index=False)
            for rk in ["A+", "A", "B", "C"]:
                df_rk = out_ios.selected_by_rank.get(rk, pd.DataFrame())
                name = f"IOS_{rk}".replace("+", "PLUS").replace("-", "DASH")
                df_rk2 = _attach_why(df_rk, out_ios.decision_log)
                _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(df_rk2), "ios")).to_excel(writer, sheet_name=name[:31], index=False)
            all_a = _attach_why(out_android.selected, out_android.decision_log)
            all_i = _attach_why(out_ios.selected, out_ios.decision_log)
            _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(all_a), "android")).to_excel(
                writer, sheet_name="AOS_Selected_All", index=False
            )
            _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(all_i), "ios")).to_excel(writer, sheet_name="IOS_Selected_All", index=False)

            if isinstance(exc_df_a, pd.DataFrame) and not exc_df_a.empty:
                _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(exc_df_a), "android")).to_excel(
                    writer, sheet_name="AOS_GPU_Exception_All", index=False
                )
            if isinstance(exc_df_i, pd.DataFrame) and not exc_df_i.empty:
                _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(exc_df_i), "ios")).to_excel(
                    writer, sheet_name="IOS_GPU_Exception_All", index=False
                )

    with download_box.container():
        st.subheader("다운로드")
        st.download_button(
            "엑셀 다운로드(멀티시트)",
            data=buf.getvalue(),
            file_name=f"{policy.project}_{version}_selection.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        # (요청) 로그 다운로드 버튼 제거(엑셀 다운로드 1개만)

        # (요청) 디버그 UI는 다운로드 버튼 아래로 이동
        st.markdown("### 디버그(필요 시)")

        # Debug: required/prev-exclude No matching against normalized data
        try:
            req_list_dbg = _split_csv_list(required_nos)
            prev_list_dbg = _split_csv_list(prev_excl_nos)
            if req_list_dbg or prev_list_dbg:
                with st.expander("디버그: 필수/직전버전 제외 No 매칭 확인", expanded=False):
                    st.write({"required_nos_input": req_list_dbg, "exclude_prev_version_nos_input": prev_list_dbg})
                    st.write({"norm_df_columns(sample)": list(norm_df.columns)[:80], "rows": int(len(norm_df))})

                    def _pick_no_col(df: pd.DataFrame) -> str:
                        for c in ["No", "NO", "no", "번호", "관리번호"]:
                            if c in df.columns:
                                return c
                        for c in df.columns:
                            if "no" in str(c).lower() or "번호" in str(c) or "관리" in str(c):
                                return str(c)
                        return ""

                    no_col = _pick_no_col(norm_df)
                    did_col = "device_id" if "device_id" in norm_df.columns else ""
                    if not no_col:
                        st.warning("norm_df에서 No 컬럼 후보를 찾지 못했습니다. (정규화 로직 확인 필요)")
                    else:
                        s_no = norm_df[no_col].fillna("").astype(str).str.strip().str.upper()
                        st.write({"no_col_detected": no_col, "device_id_col": did_col})
                        st.write({"no_preview(head)": s_no.head(20).tolist()})

                        def _match_rows(tokens: list[str]) -> pd.DataFrame:
                            if not tokens:
                                return norm_df.head(0).copy()
                            want = {str(x).strip().upper() for x in tokens if str(x).strip()}
                            m = s_no.isin(want)
                            cols = [
                                c
                                for c in [no_col, did_col, "rank", "platform", "available", "product_name", "model_name"]
                                if c and c in norm_df.columns
                            ]
                            return norm_df.loc[m, cols].copy() if cols else norm_df.loc[m].copy()

                        req_hit = _match_rows(req_list_dbg)
                        prev_hit = _match_rows(prev_list_dbg)
                        st.write({"required_hits": int(len(req_hit)), "exclude_prev_hits": int(len(prev_hit))})
                        if req_list_dbg and req_hit.empty:
                            st.error("필수 No가 업로드 데이터(norm_df)에서 0건 매칭됩니다. (No 값 자체가 없거나 다른 표기일 가능성)")
                        if prev_list_dbg and prev_hit.empty:
                            st.warning("직전 버전 제외 No가 업로드 데이터(norm_df)에서 0건 매칭됩니다. (보조 모델명 제외는 작동할 수 있음)")
                        if not req_hit.empty:
                            st.markdown("**필수 No 매칭 결과(상위 20행)**")
                            st.dataframe(req_hit.head(20), use_container_width=True)
                        if not prev_hit.empty:
                            st.markdown("**직전 버전 제외 No 매칭 결과(상위 20행)**")
                            st.dataframe(prev_hit.head(20), use_container_width=True)
        except Exception:
            pass

        # Debug: engine logs
        try:
            if isinstance(getattr(out_android, "logs", None), list) and out_android.logs:
                with st.expander("디버그: 엔진 로그(run_policy_v2) - Android", expanded=False):
                    st.text("\n".join([str(x) for x in out_android.logs[-250:]]))
        except Exception:
            pass
        try:
            if out_ios is not None and isinstance(getattr(out_ios, "logs", None), list) and out_ios.logs:
                with st.expander("디버그: 엔진 로그(run_policy_v2) - iOS", expanded=False):
                    st.text("\n".join([str(x) for x in out_ios.logs[-250:]]))
        except Exception:
            pass

        # Debug: required devices present in final outputs
        try:
            req_list_ui = _split_csv_list(st.session_state.get("required_nos", "") or "")
            req_set = {str(x).strip().upper() for x in req_list_ui if str(x).strip()}
            if req_set:
                with st.expander("디버그: 필수 디바이스가 최종 결과에 포함되었는지", expanded=False):
                    st.write({"required_nos": sorted(list(req_set))[:80]})
                    # Android
                    try:
                        sel_a = _attach_why(out_android.selected, out_android.decision_log)
                        clean_a = _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(sel_a), "android"))
                        if "No" in clean_a.columns:
                            s_no_a = clean_a["No"].fillna("").astype(str).str.strip().str.upper()
                            req_rows_a = clean_a[s_no_a.isin(req_set)].copy()
                            st.write({"android_required_in_selected": int(len(req_rows_a)), "android_selected_total": int(len(clean_a))})
                            if not req_rows_a.empty:
                                st.dataframe(req_rows_a.head(600), use_container_width=True)
                    except Exception:
                        pass
                    # iOS
                    try:
                        if out_ios is not None:
                            sel_i = _attach_why(out_ios.selected, out_ios.decision_log)
                            clean_i = _dedupe_cols(_postprocess_output_for_platform(_clean_result_df(sel_i), "ios"))
                            if "No" in clean_i.columns:
                                s_no_i = clean_i["No"].fillna("").astype(str).str.strip().str.upper()
                                req_rows_i = clean_i[s_no_i.isin(req_set)].copy()
                                st.write({"ios_required_in_selected": int(len(req_rows_i)), "ios_selected_total": int(len(clean_i))})
                                if not req_rows_i.empty:
                                    st.dataframe(req_rows_i.head(600), use_container_width=True)
                    except Exception:
                        pass
        except Exception:
            pass

        # Debug: show what got dropped by dedupe (same cpu|gpu|ram profile_core)
        def _render_dedupe_debug(out_obj: Any, *, title: str) -> None:
            try:
                dlog = getattr(out_obj, "dedupe_log", None)
                if not isinstance(dlog, pd.DataFrame) or dlog.empty:
                    return
                with st.expander(title, expanded=False):
                    st.caption(
                        "선택 로직에서 (cpu,gpu,ram_gb) 프로파일이 같은 후보는 1개만 남기고 나머지는 제외됩니다. "
                        "아래 목록이 ‘빠진 디바이스’와 일치하면, 공통 원인은 cpu/gpu/ram 중복입니다."
                    )
                    dlog_view = dlog.copy()
                    try:
                        sel = out_obj.selected.copy() if isinstance(getattr(out_obj, "selected", None), pd.DataFrame) else pd.DataFrame()
                        if not sel.empty:
                            if "rank" not in sel.columns and "Rank" in sel.columns:
                                sel = sel.rename(columns={"Rank": "rank"})
                            for k in ["cpu", "gpu", "ram_gb"]:
                                if k not in sel.columns:
                                    sel[k] = ""
                            parts = [sel[k].fillna("").astype(str).str.strip() for k in ["cpu", "gpu", "ram_gb"]]
                            prof = parts[0]
                            for p in parts[1:]:
                                prof = prof + "|" + p
                            sel["_profile_core"] = prof

                            def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
                                for c in candidates:
                                    if c in df.columns:
                                        return c
                                return ""

                            name_col = _pick_col(sel, ["product_name", "model_name", "제품명"])
                            no_col = _pick_col(sel, ["No", "NO", "no", "번호", "관리번호"])
                            did_col = _pick_col(sel, ["device_id"])
                            kept = sel[["rank", "_profile_core"]].copy()
                            kept["kept_No"] = sel[no_col].astype(str) if no_col else ""
                            kept["kept_product_name"] = sel[name_col].astype(str) if name_col else ""
                            kept["kept_device_id"] = sel[did_col].astype(str) if did_col else ""
                            kept = kept.rename(columns={"_profile_core": "profile_core"})
                            kept = kept.drop_duplicates(subset=["rank", "profile_core"], keep="first")
                            dlog_view = dlog_view.merge(kept, on=["rank", "profile_core"], how="left")
                    except Exception:
                        pass
                    st.write(
                        {
                            "dropped_rows": int(len(dlog_view)),
                            "unique_profile_core": int(dlog_view["profile_core"].nunique()) if "profile_core" in dlog_view.columns else None,
                            "reasons": dlog_view["reason"].value_counts().to_dict() if "reason" in dlog_view.columns else None,
                        }
                    )
                    if "profile_core" in dlog_view.columns:
                        grp = dlog_view.groupby("profile_core", as_index=False).size().sort_values(by="size", ascending=False)
                        st.markdown("**프로파일(core)별 제외 건수 Top**")
                        st.dataframe(grp.head(20), use_container_width=True)
                    st.markdown("**제외된 원본 목록(상위 200행)**")
                    st.dataframe(dlog_view.head(200), use_container_width=True)
            except Exception:
                return

        _render_dedupe_debug(out_android, title="디버그: 중복 제거로 제외된 후보(dedupe) - Android")
        if out_ios is not None:
            _render_dedupe_debug(out_ios, title="디버그: 중복 제거로 제외된 후보(dedupe) - iOS")

    # bottom navigation
    st.divider()
    # (요청) 결과 화면의 정책확인 버튼 제거


if __name__ == "__main__":
    _run()


