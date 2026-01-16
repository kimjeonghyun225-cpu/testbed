from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .policy_v2 import PolicyV2, normalize_df_for_policy_v2


@dataclass(frozen=True)
class PolicyV2RunInputs:
    project: str
    version: str
    rank_targets: dict[str, int]
    required_nos: list[str]
    exclude_prev_version_nos: list[str]
    exclude_target_countries: list[str]
    exclude_notes: list[str]


@dataclass(frozen=True)
class PolicyV2Outputs:
    selected: pd.DataFrame
    selected_by_rank: dict[str, pd.DataFrame]
    dedupe_log: pd.DataFrame
    shortage_summary: pd.DataFrame
    decision_log: pd.DataFrame
    rule_summary: pd.DataFrame
    logs: list[str]


RANK_ORDER = ["A+", "A", "B", "C", "D", "-"]


def _norm_rank(x: object) -> str:
    s = str(x or "").strip().upper().replace(" ", "")
    if s in ("A+", "A", "B", "C", "D", "-"):
        return s
    return "-" if s else ""


def _gpu_family(gpu: object) -> str:
    s = str(gpu or "").strip().lower()
    if not s:
        return "unknown"
    if "adreno" in s:
        return "adreno"
    if "immortalis" in s:
        return "immortalis"
    if "mali" in s:
        return "mali"
    if "powervr" in s:
        return "powervr"
    return "other"


def _pick_diversity_must_cover(
    df: pd.DataFrame,
    *,
    must_cover_axes: tuple[str, ...],
    target_n: int,
) -> pd.DataFrame:
    """
    must_cover_axes가 있을 때, 가능한 한 다양한 축 값을 먼저 커버하도록 후보를 재정렬한다.
    - greedy: 다음 후보 선택 시 "새로 커버되는 축 값 개수"가 최대인 항목을 우선
    - tie: 원래 순서(_base_ord) 유지
    """
    if df is None or df.empty or target_n <= 0 or not must_cover_axes:
        return df

    tmp = df.copy()
    tmp["_base_ord"] = range(len(tmp))
    for ax in must_cover_axes:
        if ax not in tmp.columns:
            tmp[ax] = ""
        tmp[ax] = tmp[ax].fillna("").astype(str).str.strip().str.lower()

    seen: dict[str, set[str]] = {ax: set() for ax in must_cover_axes}
    picked_idx: list[int] = []
    remain = list(tmp.index)

    def _gain(ix: int) -> tuple[int, int, str]:
        gain = 0
        for ax in must_cover_axes:
            v = str(tmp.at[ix, ax] or "")
            if v and v not in seen[ax]:
                gain += 1
        base = int(tmp.at[ix, "_base_ord"])
        did = str(tmp.at[ix, "device_id"]) if "device_id" in tmp.columns else str(ix)
        # negative gain for max selection via min()
        return (-gain, base, did)

    while remain and len(picked_idx) < min(target_n, len(tmp)):
        best = min(remain, key=_gain)
        picked_idx.append(best)
        for ax in must_cover_axes:
            v = str(tmp.at[best, ax] or "")
            if v:
                seen[ax].add(v)
        remain.remove(best)

    # append rest in original order
    rest = tmp.loc[remain].sort_values(by="_base_ord")
    front = tmp.loc[picked_idx] if picked_idx else tmp.head(0)
    out = pd.concat([front, rest], axis=0)
    return out.drop(columns=["_base_ord"], errors="ignore")

def _manufacturer_soft_dedupe_params(policy: PolicyV2) -> tuple[bool, float]:
    """
    manufacturer_policy 기반으로 제조사 soft_dedupe 사용 여부/가중치를 반환.
    (하위호환) 기존 tie_breakers의 brand_penalty_after_profile_coverage도 인식한다.
    """
    # new schema first
    if policy.manufacturer_policy and policy.manufacturer_policy.within_rank:
        mode = str(policy.manufacturer_policy.within_rank.mode or "").strip().lower()
        if mode == "soft_dedupe":
            try:
                w = float(policy.manufacturer_policy.within_rank.penalty_weight or 0.0)
            except Exception:
                w = 0.0
            return (True, max(0.0, w))
        return (False, 0.0)

    # backward compatibility: previous implementation used tie_breakers dict
    for tb in policy.tie_breakers:
        if isinstance(tb, dict) and tb.get("brand_penalty_after_profile_coverage") is True:
            return (True, 1.0)
    return (False, 0.0)


def _get_brand(v: object) -> str:
    s = str(v or "").strip()
    return s


def _pick_brand_soft_dedupe(
    df: pd.DataFrame, *, target_n: int, penalty_weight: float
) -> pd.DataFrame:
    """
    soft_dedupe: base 정렬 순서를 유지하되, 이미 선택된 제조사(brand)가 많을수록 패널티를 부여해
    다른 제조사가 섞이도록 greedy로 선택한다.

    score = base_order + penalty_weight * brand_used_count[brand]
    (penalty_weight=0이면 base_order 그대로)
    """
    if df is None or df.empty or target_n <= 0 or "brand" not in df.columns:
        return df.head(0) if target_n <= 0 else df
    tmp = df.copy()
    tmp["_base_ord"] = range(len(tmp))
    brand_counts: dict[str, int] = {}
    picked: list[int] = []
    remaining = list(tmp.index)

    def _score(ix: int) -> tuple[float, int, str]:
        b = _get_brand(tmp.at[ix, "brand"])
        bc = brand_counts.get(b, 0) if b else 0
        base = int(tmp.at[ix, "_base_ord"])
        did = str(tmp.at[ix, "device_id"]) if "device_id" in tmp.columns else str(ix)
        return (base + float(penalty_weight) * float(bc), base, did)

    while remaining and len(picked) < target_n:
        best = min(remaining, key=_score)
        picked.append(best)
        b = _get_brand(tmp.at[best, "brand"])
        if b:
            brand_counts[b] = brand_counts.get(b, 0) + 1
        remaining.remove(best)

    out = tmp.loc[picked].copy()
    return out.drop(columns=["_base_ord"], errors="ignore")


def _brand_diverse_fill(
    candidates: pd.DataFrame,
    *,
    selected_rows: list[pd.Series],
    target_n: int,
    used_ids: set[str],
    core_counts: dict[str, int],
    max_dup: int,
    penalty_weight: float = 1.0,
    require_core_col: str = "_profile_core",
    can_take: Optional[callable] = None,
    on_take: Optional[callable] = None,
) -> None:
    """
    남은 후보를 selected_rows에 채운다.
    - 대표성(프로파일 제약)은 호출자가 core_counts/max_dup로 제어
    - brand_penalty_after_profile_coverage 목적: 이미 선택된 제조사는 뒤로 밀어준다(강제는 아님)
    """
    if candidates is None or candidates.empty or len(selected_rows) >= target_n:
        return
    cand = candidates.copy()
    if "brand" not in cand.columns:
        # brand 정보가 없으면 기존 순서대로 채움
        for _, rr in cand.iterrows():
            if len(selected_rows) >= target_n:
                break
            core = str(rr.get(require_core_col) or "")
            if core_counts.get(core, 0) >= max_dup:
                continue
            did = str(rr.get("device_id") or "")
            if did and did not in used_ids:
                selected_rows.append(rr)
                used_ids.add(did)
                core_counts[core] = core_counts.get(core, 0) + 1
        return

    used_brands = {_get_brand(r.get("brand")) for r in selected_rows if _get_brand(r.get("brand"))}
    # greedy: score = base_order + penalty_weight * brand_used_count
    cand["_base_ord"] = range(len(cand))
    brand_counts: dict[str, int] = {b: 1 for b in used_brands if b}

    def _score_row(rr: pd.Series) -> tuple[float, int, str]:
        b = _get_brand(rr.get("brand"))
        bc = brand_counts.get(b, 0) if b else 0
        base = int(rr.get("_base_ord") or 0)
        did = str(rr.get("device_id") or "")
        return (base + float(penalty_weight) * float(bc), base, did)

    remain = [rr for _, rr in cand.iterrows()]
    while remain and len(selected_rows) < target_n:
        best_rr = min(remain, key=_score_row)
        core = str(best_rr.get(require_core_col) or "")
        if can_take is not None:
            try:
                if not bool(can_take(best_rr)):
                    remain.remove(best_rr)
                    continue
            except Exception:
                pass
        if core_counts.get(core, 0) < max_dup:
            did = str(best_rr.get("device_id") or "")
            if did and did not in used_ids:
                selected_rows.append(best_rr)
                used_ids.add(did)
                if on_take is not None:
                    try:
                        on_take(best_rr)
                    except Exception:
                        pass
                b = _get_brand(best_rr.get("brand"))
                if b:
                    brand_counts[b] = brand_counts.get(b, 0) + 1
                core_counts[core] = core_counts.get(core, 0) + 1
        remain.remove(best_rr)


def _tie_key(df: pd.DataFrame, tie_breakers: tuple[object, ...]) -> pd.DataFrame:
    """
    tie_breakers에 따라 정렬 키 컬럼을 생성한다(결정 가능한 데이터만).
    반환된 DF는 정렬에 필요한 보조 컬럼을 포함할 수 있다.
    """
    out = df.copy()
    out["_gpu_family"] = out["gpu"].map(_gpu_family) if "gpu" in out.columns else "unknown"
    # display resolution score
    if "display_w" in out.columns and "display_h" in out.columns:
        w = pd.to_numeric(out["display_w"], errors="coerce").fillna(0)
        h = pd.to_numeric(out["display_h"], errors="coerce").fillna(0)
        out["_disp_score"] = (w * h).astype("int64")
    else:
        out["_disp_score"] = 0

    # release year
    if "release_year" in out.columns:
        out["_year"] = pd.to_numeric(out["release_year"], errors="coerce").fillna(0).astype(int)
    else:
        out["_year"] = 0

    # No prefix priority (e.g., android: AP>AT>R, ios: IP>IT>R)
    out["_no_pri"] = 999
    for tb in tie_breakers:
        if isinstance(tb, dict) and "no_prefix_priority" in tb and isinstance(tb["no_prefix_priority"], dict):
            cfg = tb.get("no_prefix_priority") or {}
            a_list = [str(x).strip().upper() for x in (cfg.get("android") or ["AP", "AT", "R"]) if str(x).strip()]
            i_list = [str(x).strip().upper() for x in (cfg.get("ios") or ["IP", "IT", "R"]) if str(x).strip()]
            a_mp = {p: i for i, p in enumerate(a_list)}
            i_mp = {p: i for i, p in enumerate(i_list)}

            if "No" in out.columns:
                s_no = out["No"].fillna("").astype(str).str.strip()
                # prefix: leading letters 1~2 (e.g., AP632->AP, R001->R, iT123->IT)
                pref = s_no.str.extract(r"(?i)^([a-z]{1,2})", expand=False).fillna("").astype(str).str.upper()
            else:
                pref = pd.Series([""] * len(out), index=out.index)

            if "platform" in out.columns:
                plat = out["platform"].fillna("").astype(str).str.strip().str.lower()
            else:
                plat = pd.Series(["android"] * len(out), index=out.index)

            def _pri(pfx: str, p: str) -> int:
                if p == "ios":
                    return int(i_mp.get(pfx, 999))
                return int(a_mp.get(pfx, 999))

            out["_no_pri"] = pd.concat([pref, plat], axis=1).apply(lambda r: _pri(str(r[0] or ""), str(r[1] or "")), axis=1).astype(int)
            break

    # gpu family priority
    # default: unknown last
    out["_gpu_pri"] = 999
    for tb in tie_breakers:
        if isinstance(tb, dict) and "gpu_family_priority" in tb and isinstance(tb["gpu_family_priority"], list):
            pri = [str(x).strip().lower() for x in tb["gpu_family_priority"] if str(x).strip()]
            mp = {name: i for i, name in enumerate(pri)}
            out["_gpu_pri"] = out["_gpu_family"].map(lambda x: mp.get(str(x).lower(), 999)).astype(int)
            break
    return out


def _sort_for_pick(df: pd.DataFrame, tie_breakers: tuple[object, ...]) -> pd.DataFrame:
    """
    deterministic sort for representative selection
    """
    tmp = _tie_key(df, tie_breakers)
    cols: list[str] = []
    asc: list[bool] = []
    # apply known tie breakers in order
    for tb in tie_breakers:
        if isinstance(tb, dict) and "no_prefix_priority" in tb:
            cols.append("_no_pri"); asc.append(True)
        elif tb == "newest_release_year":
            cols.append("_year"); asc.append(False)
        elif tb == "higher_display_resolution":
            cols.append("_disp_score"); asc.append(False)
        elif isinstance(tb, dict) and "gpu_family_priority" in tb:
            cols.append("_gpu_pri"); asc.append(True)
    # fallback deterministic
    if "device_id" in tmp.columns:
        cols.append("device_id"); asc.append(True)
    elif "No" in tmp.columns:
        cols.append("No"); asc.append(True)
    return tmp.sort_values(by=cols, ascending=asc) if cols else tmp


def _apply_candidate_filter(df: pd.DataFrame, policy: PolicyV2, logs: list[str]) -> pd.DataFrame:
    out = df.copy()

    # platform filter (android/ios/both)
    plat_raw = str(getattr(policy.candidate_filter, "platform", "") or "").strip().lower()
    if plat_raw:
        s = plat_raw.replace("|", ",").replace(";", ",").replace(" ", ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        plats: set[str] = set()
        for p in parts:
            if p in ("android", "aos"):
                plats.add("android")
            elif p in ("ios", "iphone", "ipad"):
                plats.add("ios")
            elif p in ("both", "all", "*"):
                plats = {"android", "ios"}
        if plats and plats != {"android", "ios"}:
            if "platform" in out.columns:
                out["_platform_norm"] = out["platform"].fillna("").astype(str).str.strip().str.lower().replace({"aos": "android"})
                before = int(len(out))
                out = out[out["_platform_norm"].isin(plats)].copy()
                after = int(len(out))
                logs.append(f"[platform_filter] platforms={sorted(plats)} kept={after} dropped={before-after}")
                out = out.drop(columns=["_platform_norm"], errors="ignore")
            else:
                logs.append("[warn] candidate_filter.platform set but platform column missing; skip")

    # availability usable_only
    if policy.candidate_filter.availability.usable_only:
        if "available" in out.columns:
            out = out[out["available"].fillna(False).astype(bool)].copy()
        else:
            logs.append("[warn] candidate_filter.usable_only=true but available column missing; skip")

    # rentable_only best-effort
    if policy.candidate_filter.availability.rentable_only:
        rent_cols = [c for c in ["대여가능여부", "rentable", "raw__대여가능여부"] if c in out.columns]
        if rent_cols:
            rc = rent_cols[0]
            s = out[rc].fillna("").astype(str).str.strip().str.lower()
            ok = s.isin(["1", "true", "y", "yes", "가능", "대여가능", "ok"])
            out = out[ok].copy()
        else:
            logs.append("[warn] candidate_filter.rentable_only=true but no rentable column; skip")

    # required fields
    req = set(policy.candidate_filter.required_fields or ())
    missing = [f for f in req if f not in out.columns]
    if missing:
        logs.append(f"[warn] required_fields missing columns={missing} (will reduce candidates)")
    for f in req:
        if f not in out.columns:
            continue
        if f in ("ram_gb", "display_w", "display_h", "release_year"):
            out = out[pd.to_numeric(out[f], errors="coerce").notna()].copy()
        else:
            out = out[out[f].fillna("").astype(str).str.strip().ne("")].copy()

    # numeric ranges (inclusive)
    nr = getattr(policy.candidate_filter, "numeric_ranges", {}) or {}
    if isinstance(nr, dict) and nr:
        for col, rng in nr.items():
            if col not in out.columns:
                logs.append(f"[warn] candidate_filter.numeric_ranges set but missing column={col}; skip")
                continue
            try:
                lo, hi = float(rng[0]), float(rng[1])
            except Exception:
                logs.append(f"[warn] candidate_filter.numeric_ranges invalid range for {col}: {rng}; skip")
                continue
            before = int(len(out))
            s = pd.to_numeric(out[col], errors="coerce")
            keep = s.notna() & (s >= lo) & (s <= hi)
            out = out[keep].copy()
            after = int(len(out))
            logs.append(f"[numeric_ranges] {col} in [{lo}, {hi}] kept={after} dropped={before-after}")

    return out


def _map_no_to_model_names(df: pd.DataFrame, nos: list[str]) -> tuple[set[str], list[str]]:
    """
    No -> product_name/model_name mapping from current df
    """
    tokens = [str(x).strip() for x in nos if str(x).strip()]
    if not tokens:
        return (set(), [])
    if "No" not in df.columns:
        return (set(), tokens)
    name_cols = [c for c in ["product_name", "model_name", "제품명", "모델명"] if c in df.columns]
    if not name_cols:
        return (set(), tokens)
    tmp = df[["No"] + name_cols].copy()
    tmp["No"] = tmp["No"].fillna("").astype(str).str.strip()
    for c in name_cols:
        tmp[c] = tmp[c].fillna("").astype(str).str.strip()
    no_to_names: dict[str, set[str]] = {}
    for _, r in tmp.iterrows():
        no = str(r["No"]).strip()
        if not no:
            continue
        ss = {str(r[c]).strip() for c in name_cols if str(r[c]).strip()}
        if ss:
            no_to_names.setdefault(no, set()).update(ss)
    names: set[str] = set()
    unresolved: list[str] = []
    for t in tokens:
        ns = no_to_names.get(t)
        if ns:
            names |= ns
        else:
            unresolved.append(t)
    # normalize
    names2 = {str(x).strip().upper() for x in names if str(x).strip()}
    return (names2, unresolved)


def run_policy_v2(
    *,
    master_df: pd.DataFrame,
    policy: PolicyV2,
    inputs: PolicyV2RunInputs,
) -> PolicyV2Outputs:
    logs: list[str] = []
    # global stats for explainability (used in why_detail)
    stats: dict[str, Any] = {
        "master_rows": int(len(master_df)) if master_df is not None else 0,
        "after_normalize_rows": 0,
        "after_candidate_filter_rows": 0,
        "exclude_target_countries_enabled": False,
        "exclude_target_countries_targets": 0,
        "exclude_target_countries_dropped": 0,
        "exclude_target_countries_kept_required": 0,
        "exclude_notes_enabled": False,
        "exclude_notes_targets": 0,
        "exclude_notes_dropped": 0,
        "exclude_notes_kept_required": 0,
        "prev_version_exclude_enabled": False,
        "prev_version_exclude_model_names": 0,
        "prev_version_exclude_dropped": 0,
        "prev_version_exclude_kept_required": 0,
        "prev_version_exclude_unresolved_nos": 0,
    }

    # normalize to canonical columns
    df0 = normalize_df_for_policy_v2(master_df)
    stats["after_normalize_rows"] = int(len(df0))
    if "rank" in df0.columns:
        df0["rank"] = df0["rank"].map(_norm_rank)

    # candidate filter
    df = _apply_candidate_filter(df0, policy, logs)
    stats["after_candidate_filter_rows"] = int(len(df))

    # required devices by No: force include
    required_nos = [str(x).strip() for x in inputs.required_nos if str(x).strip()]
    required_ids: set[str] = set()
    if required_nos and "No" in df.columns:
        tmp = df[["No", "device_id"]].copy() if "device_id" in df.columns else df[["No"]].copy()
        tmp["No"] = tmp["No"].fillna("").astype(str).str.strip()
        if "device_id" in tmp.columns:
            tmp["device_id"] = tmp["device_id"].fillna("").astype(str).str.strip()
        no_to_id = {str(r["No"]): str(r.get("device_id") or "") for _, r in tmp.iterrows() if str(r["No"]).strip()}
        for n in required_nos:
            did = no_to_id.get(n, "")
            if did:
                required_ids.add(did)
    elif required_nos and "device_id" in df.columns:
        required_ids |= set(required_nos)

    # exclude target countries (optional)
    excl_targets = [str(x).strip() for x in (inputs.exclude_target_countries or []) if str(x).strip()]
    if excl_targets:
        stats["exclude_target_countries_enabled"] = True
        excl_set = {str(x).strip().lower() for x in excl_targets if str(x).strip()}
        stats["exclude_target_countries_targets"] = int(len(excl_set))
        col = "target_market" if "target_market" in df.columns else ("타겟 국가" if "타겟 국가" in df.columns else None)
        if col:
            s = df[col].fillna("").astype(str).str.strip().str.lower()
            drop_m = s.isin(excl_set)
            if required_ids and "device_id" in df.columns:
                keep_req = df["device_id"].fillna("").astype(str).isin(required_ids)
            elif required_nos and "No" in df.columns:
                keep_req = df["No"].fillna("").astype(str).str.strip().isin(set(required_nos))
            else:
                keep_req = pd.Series([False] * len(df), index=df.index)
            dropped = int((drop_m & (~keep_req)).sum())
            kept_req = int((drop_m & keep_req).sum())
            stats["exclude_target_countries_dropped"] = dropped
            stats["exclude_target_countries_kept_required"] = kept_req
            if dropped:
                df = df[~(drop_m & (~keep_req))].copy()
            logs.append(f"[exclude_target_countries] targets={len(excl_set)} dropped={dropped} kept_required={kept_req}")
        else:
            logs.append("[warn] exclude_target_countries set but target_market/타겟 국가 column missing; skip")

    # exclude notes (best-effort): drop rows with matching note text, but keep required
    excl_notes = [str(x).strip() for x in (inputs.exclude_notes or []) if str(x).strip()]
    if excl_notes:
        stats["exclude_notes_enabled"] = True
        excl_set = {str(x).strip().lower() for x in excl_notes if str(x).strip()}
        stats["exclude_notes_targets"] = int(len(excl_set))
        col = "note" if "note" in df.columns else ("NOTE" if "NOTE" in df.columns else ("raw__NOTE" if "raw__NOTE" in df.columns else None))
        if col:
            s = df[col].fillna("").astype(str).str.strip().str.lower()
            drop_m = s.isin(excl_set)
            if required_ids and "device_id" in df.columns:
                keep_req = df["device_id"].fillna("").astype(str).isin(required_ids)
            elif required_nos and "No" in df.columns:
                keep_req = df["No"].fillna("").astype(str).str.strip().isin(set(required_nos))
            else:
                keep_req = pd.Series([False] * len(df), index=df.index)
            dropped = int((drop_m & (~keep_req)).sum())
            kept_req = int((drop_m & keep_req).sum())
            stats["exclude_notes_dropped"] = dropped
            stats["exclude_notes_kept_required"] = kept_req
            if dropped:
                df = df[~(drop_m & (~keep_req))].copy()
            logs.append(f"[exclude_notes] targets={len(excl_set)} dropped={dropped} kept_required={kept_req}")
        else:
            logs.append("[warn] exclude_notes set but note/NOTE column missing; skip")

    # exclude previous version models by No -> model names
    excl_names, unresolved_excl = _map_no_to_model_names(df, inputs.exclude_prev_version_nos)
    if inputs.exclude_prev_version_nos:
        stats["prev_version_exclude_enabled"] = True
    stats["prev_version_exclude_unresolved_nos"] = int(len(unresolved_excl))
    if unresolved_excl:
        logs.append(f"[warn] prev_version_exclude unresolved No={unresolved_excl[:20]}")
    if excl_names:
        stats["prev_version_exclude_model_names"] = int(len(excl_names))
        name_cols = [c for c in ["product_name", "model_name", "제품명", "모델명"] if c in df.columns]
        if name_cols:
            m_any = pd.Series([False] * len(df), index=df.index)
            for c in name_cols:
                m_any = m_any | df[c].fillna("").astype(str).str.strip().str.upper().isin(excl_names)
            # keep required_ids
            if required_ids and "device_id" in df.columns:
                keep_req = df["device_id"].fillna("").astype(str).isin(required_ids)
            else:
                keep_req = pd.Series([False] * len(df), index=df.index)
            drop_m = m_any & (~keep_req)
            dropped = int(drop_m.sum())
            kept_req = int((m_any & keep_req).sum())
            stats["prev_version_exclude_dropped"] = dropped
            stats["prev_version_exclude_kept_required"] = kept_req
            if dropped:
                df = df[~drop_m].copy()
            logs.append(f"[prev_version_exclude] model_names={len(excl_names)} dropped={dropped} kept_required={kept_req}")

    # per-rank selection
    dedupe_key = tuple(policy.dedupe.within_rank.key or ())
    allow_axes = tuple(policy.dedupe.within_rank.allow_duplicates_if_differs_by or ())
    allow_duplicates = bool(policy.dedupe.within_rank.allow_duplicates)
    max_dup = int(policy.dedupe.within_rank.max_duplicates_per_profile or 1)
    max_dup = max(1, max_dup)
    # allow_axes가 있는 경우(대표성 확보 후에만 허용), 최소 2까지는 열어준다
    if allow_axes and (not allow_duplicates) and max_dup < 2:
        max_dup = 2

    must_cover_axes = tuple(policy.diversity.within_rank.must_cover or ())
    must_include_if_available = getattr(policy.diversity.within_rank, "must_include_if_available", {}) or {}
    max_per_product_name = int(getattr(policy.dedupe.within_rank, "max_per_product_name", 0) or 0)
    use_manu, manu_w = _manufacturer_soft_dedupe_params(policy)

    selected_by_rank: dict[str, pd.DataFrame] = {}
    dedupe_rows: list[dict[str, Any]] = []
    shortage_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for rk in RANK_ORDER:
        target_n = int(inputs.rank_targets.get(rk, 0) or 0)
        if target_n <= 0:
            selected_by_rank[rk] = df.head(0).copy()
            shortage_rows.append({"rank": rk, "target": target_n, "selected": 0, "shortage": 0, "reason": "skipped"})
            continue
        sub = df[df["rank"].fillna("").astype(str) == rk].copy()
        if sub.empty:
            shortage_rows.append({"rank": rk, "target": target_n, "selected": 0, "shortage": target_n, "reason": "후보 없음"})
            selected_by_rank[rk] = df.head(0).copy()
            continue

        # build profile core id (dedupe_key only)
        for k in dedupe_key:
            if k not in sub.columns:
                sub[k] = ""
        parts = [sub[k].fillna("").astype(str).str.strip() for k in dedupe_key]
        prof = parts[0]
        for p in parts[1:]:
            prof = prof + "|" + p
        sub["_profile_core"] = prof

        # allow_axes key (extras selection에서만 사용)
        if allow_axes:
            for ax in allow_axes:
                if ax not in sub.columns:
                    sub[ax] = ""
            axp = sub[allow_axes[0]].fillna("").astype(str).str.strip()
            for a in allow_axes[1:]:
                axp = axp + "|" + sub[a].fillna("").astype(str).str.strip()
            sub["_axes_key"] = axp
        else:
            sub["_axes_key"] = ""

        # pick representatives
        sub_sorted = _sort_for_pick(sub, policy.tie_breakers)
        # rank-level stats for why_detail
        rank_candidates = int(len(sub_sorted))
        rank_unique_profiles = int(sub_sorted["_profile_core"].nunique()) if "_profile_core" in sub_sorted.columns else 0
        axis_universe: dict[str, list[str]] = {}
        for ax in must_cover_axes:
            if ax in sub_sorted.columns:
                vals = (
                    sub_sorted[ax]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .tolist()
                )
                axis_universe[ax] = sorted({v for v in vals if v})
            else:
                axis_universe[ax] = []
        # always include required devices in this rank (if present)
        req_in_rank = set()
        if required_ids and "device_id" in sub_sorted.columns:
            req_in_rank = set(sub_sorted[sub_sorted["device_id"].fillna("").astype(str).isin(required_ids)]["device_id"].astype(str).tolist())

        # required selection first (do not drop even if it exceeds target)
        selected_rows: list[pd.Series] = []
        used_ids: set[str] = set()
        core_counts: dict[str, int] = {}
        core_axes_seen: dict[str, set[str]] = {}
        # product_name cap tracking (within rank)
        prod_counts: dict[str, int] = {}
        # diversity coverage tracking (within rank)
        covered: dict[str, set[str]] = {ax: set() for ax in must_cover_axes}

        def _prod_key(rr: pd.Series) -> str:
            for c in ["product_name", "제품명", "model_name"]:
                if c in rr.index:
                    s = str(rr.get(c) or "").strip()
                    if s:
                        return s.upper()
            return ""

        def _can_take(rr: pd.Series, *, is_required: bool) -> bool:
            if is_required:
                return True
            if max_per_product_name <= 0:
                return True
            pk = _prod_key(rr)
            if not pk:
                return True
            return int(prod_counts.get(pk, 0)) < max_per_product_name

        def _mark_taken(rr: pd.Series) -> None:
            if max_per_product_name <= 0:
                return
            pk = _prod_key(rr)
            if not pk:
                return
            prod_counts[pk] = int(prod_counts.get(pk, 0)) + 1

        if req_in_rank and "device_id" in sub_sorted.columns:
            req_df = sub_sorted[sub_sorted["device_id"].fillna("").astype(str).isin(req_in_rank)].copy()
            for _, rr in req_df.iterrows():
                did = str(rr.get("device_id") or "")
                if did and did not in used_ids:
                    selected_rows.append(rr)
                    used_ids.add(did)
                    _mark_taken(rr)
                    core = str(rr.get("_profile_core") or "")
                    core_counts[core] = core_counts.get(core, 0) + 1
                    axk = str(rr.get("_axes_key") or "")
                    core_axes_seen.setdefault(core, set()).add(axk)
                    # mark covered axes
                    for ax in must_cover_axes:
                        v = str(rr.get(ax) or "").strip().lower()
                        if v:
                            covered.setdefault(ax, set()).add(v)
                    # coverage view for this decision row
                    cov_lines: list[str] = []
                    for ax in must_cover_axes:
                        uni = axis_universe.get(ax, [])
                        cov = sorted(covered.get(ax, set()))
                        remain = [x for x in uni if x not in set(cov)]
                        cov_lines.append(
                            f"- cover({ax}): {len(cov)}/{len(uni)} "
                            f"(now={cov[:8]}{'...' if len(cov) > 8 else ''}, "
                            f"remain={remain[:8]}{'...' if len(remain) > 8 else ''})"
                        )
                    # tie-breaker key values (best-effort)
                    yv = rr.get("_year")
                    dv = rr.get("_disp_score")
                    gp = rr.get("_gpu_pri")
                    decision_rows.append(
                        {
                            "rank": rk,
                            "device_id": did,
                            "No": str(rr.get("No") or ""),
                            "product_name": str(rr.get("product_name") or rr.get("model_name") or ""),
                            "brand": str(rr.get("brand") or ""),
                            "cpu_family": str(rr.get("cpu_family") or ""),
                            "profile_core": core,
                            "axes_key": axk,
                            "stage": "required",
                            "why": "필수 디바이스로 지정되어 포함",
                            "why_detail": (
                                f"[필수] Rank={rk} 필수 No/ID 지정으로 무조건 포함\n"
                                f"- rank_candidates(after_filter/excl)={rank_candidates}, unique_profiles={rank_unique_profiles}\n"
                                f"- exclude_target_countries: enabled={bool(stats['exclude_target_countries_enabled'])} "
                                f"targets={int(stats['exclude_target_countries_targets'])} dropped={int(stats['exclude_target_countries_dropped'])} kept_required={int(stats['exclude_target_countries_kept_required'])}\n"
                                f"- exclude_notes: enabled={bool(stats['exclude_notes_enabled'])} "
                                f"targets={int(stats['exclude_notes_targets'])} dropped={int(stats['exclude_notes_dropped'])} kept_required={int(stats['exclude_notes_kept_required'])}\n"
                                f"- prev_version_exclude: enabled={bool(stats['prev_version_exclude_enabled'])} "
                                f"model_names={int(stats['prev_version_exclude_model_names'])} dropped={int(stats['prev_version_exclude_dropped'])} "
                                f"kept_required={int(stats['prev_version_exclude_kept_required'])} unresolved_nos={int(stats['prev_version_exclude_unresolved_nos'])}\n"
                                f"- profile_core={core}\n"
                                f"- cpu_family={str(rr.get('cpu_family') or '')}\n"
                                f"- tie_keys: year={yv} disp_score={dv} gpu_pri={gp}\n"
                                + ("\n".join(cov_lines) + "\n" if cov_lines else "")
                                + f"- (참고) 제외 조건/다양성/중복 제거보다 우선"
                            ),
                        }
                    )

        # must_include_if_available (e.g., PAD): if any candidate exists, pick one early
        if target_n > 0 and len(selected_rows) < target_n and isinstance(must_include_if_available, dict) and must_include_if_available:
            for ax, want_vals in must_include_if_available.items():
                if not want_vals or ax not in sub_sorted.columns:
                    continue
                s_ax = sub_sorted[ax].fillna("").astype(str).str.strip().str.lower()
                for want in want_vals:
                    want = str(want or "").strip().lower()
                    if not want:
                        continue
                    cand = sub_sorted[(s_ax == want)].copy()
                    if cand.empty:
                        continue
                    if "device_id" in cand.columns and used_ids:
                        cand = cand[~cand["device_id"].fillna("").astype(str).isin(used_ids)].copy()
                    picked_rr: Optional[pd.Series] = None
                    for _, rr in cand.iterrows():
                        if not _can_take(rr, is_required=False):
                            continue
                        picked_rr = rr
                        break
                    if picked_rr is None:
                        continue
                    did = str(picked_rr.get("device_id") or "")
                    if did and did not in used_ids:
                        selected_rows.append(picked_rr)
                        used_ids.add(did)
                        _mark_taken(picked_rr)
                        core = str(picked_rr.get("_profile_core") or "")
                        core_counts[core] = core_counts.get(core, 0) + 1
                        axk = str(picked_rr.get("_axes_key") or "")
                        core_axes_seen.setdefault(core, set()).add(axk)
                        for ax2 in must_cover_axes:
                            v2 = str(picked_rr.get(ax2) or "").strip().lower()
                            if v2:
                                covered.setdefault(ax2, set()).add(v2)
                        decision_rows.append(
                            {
                                "rank": rk,
                                "device_id": did,
                                "No": str(picked_rr.get("No") or ""),
                                "product_name": str(picked_rr.get("product_name") or picked_rr.get("model_name") or ""),
                                "brand": str(picked_rr.get("brand") or ""),
                                "cpu_family": str(picked_rr.get("cpu_family") or ""),
                                "profile_core": str(picked_rr.get("_profile_core") or ""),
                                "axes_key": str(picked_rr.get("_axes_key") or ""),
                                "stage": "must_include",
                                "why": f"가능하면 포함: {ax}={want}",
                                "why_detail": f"[must_include] Rank={rk} {ax}={want} 후보가 있어 1대 우선 포함",
                            }
                        )
                    # only one per axis value
                    break

        # 1) core representatives first (unique by profile_core)
        reps = []
        for pid, g in sub_sorted.groupby("_profile_core", sort=False):
            g2 = g.copy()
            # if any required in this profile, keep required device first
            if req_in_rank and "device_id" in g2.columns:
                g2["_is_req"] = g2["device_id"].fillna("").astype(str).isin(req_in_rank)
                g2 = g2.sort_values(by=["_is_req"], ascending=[False])
            # 대표 후보(첫 행)를 reps로 (제품명 cap이 켜져 있으면 가능한 대표를 찾는다)
            picked = None
            for _, rr in g2.iterrows():
                if _can_take(rr, is_required=False):
                    picked = rr
                    break
            reps.append(picked if picked is not None else g2.iloc[0])
            # dedupe log for dropped (same core)
            drop = g2.iloc[1:]
            if not drop.empty:
                for _, rr in drop.iterrows():
                    dedupe_rows.append(
                        {
                            "rank": rk,
                            "dropped_device_id": str(rr.get("device_id") or ""),
                            "dropped_No": str(rr.get("No") or ""),
                            "dropped_product_name": str(rr.get("product_name") or rr.get("model_name") or ""),
                            "reason": "duplicate_profile_core",
                            "profile_core": str(pid),
                        }
                    )

        reps_df = pd.DataFrame(reps) if reps else sub_sorted.head(0).copy()

        # diversity must_cover (e.g., cpu_family): reorder core reps for coverage first
        if must_cover_axes and not reps_df.empty:
            reps_df = _pick_diversity_must_cover(reps_df, must_cover_axes=must_cover_axes, target_n=target_n)

        # manufacturer soft_dedupe: apply after profile coverage ordering
        if use_manu and not reps_df.empty:
            reps_df = _pick_brand_soft_dedupe(reps_df, target_n=target_n, penalty_weight=manu_w)

        # core reps selection (after required already included)
        if not reps_df.empty:
            for _, r in reps_df.iterrows():
                if len(selected_rows) >= target_n and (not req_in_rank):
                    break
                did = str(r.get("device_id") or "")
                if did and did not in used_ids:
                    if not _can_take(r, is_required=False):
                        continue
                    # coverage before update (for progress logging)
                    covered_before = {ax: set(covered.get(ax, set())) for ax in must_cover_axes}
                    selected_rows.append(r)
                    used_ids.add(did)
                    _mark_taken(r)
                    core = str(r.get("_profile_core") or "")
                    core_counts[core] = core_counts.get(core, 0) + 1
                    axk = str(r.get("_axes_key") or "")
                    core_axes_seen.setdefault(core, set()).add(axk)
                    newly: list[str] = []
                    for ax in must_cover_axes:
                        v = str(r.get(ax) or "").strip().lower()
                        if v and v not in covered.setdefault(ax, set()):
                            newly.append(f"{ax}={v}")
                            covered[ax].add(v)
                    cov_lines = []
                    for ax in must_cover_axes:
                        uni = axis_universe.get(ax, [])
                        cov_b = sorted(covered_before.get(ax, set()))
                        cov_a = sorted(covered.get(ax, set()))
                        remain = [x for x in uni if x not in set(cov_a)]
                        cov_lines.append(
                            f"- cover({ax}): {len(cov_b)}/{len(uni)} -> {len(cov_a)}/{len(uni)} "
                            f"(added={newly if newly else []}, remain={remain[:8]}{'...' if len(remain) > 8 else ''})"
                        )
                    yv = r.get("_year")
                    dv = r.get("_disp_score")
                    gp = r.get("_gpu_pri")
                    decision_rows.append(
                        {
                            "rank": rk,
                            "device_id": did,
                            "No": str(r.get("No") or ""),
                            "product_name": str(r.get("product_name") or r.get("model_name") or ""),
                            "brand": str(r.get("brand") or ""),
                            "cpu_family": str(r.get("cpu_family") or ""),
                            "profile_core": core,
                            "axes_key": axk,
                            "stage": "core_rep",
                            "why": "프로파일 대표(core_rep)로 선정",
                            "why_detail": (
                                f"[core_rep] Rank={rk} 목표={target_n}대 충족을 위해 (cpu,gpu,ram) 프로파일 대표로 선정\n"
                                f"- rank_candidates(after_filter/excl)={rank_candidates}, unique_profiles={rank_unique_profiles}\n"
                                f"- selection_progress: selected={len(selected_rows)}/{target_n} (including required)\n"
                                f"- profile_core={core}\n"
                                f"- cpu_family={str(r.get('cpu_family') or '')}\n"
                                f"- tie_keys: year={yv} disp_score={dv} gpu_pri={gp}\n"
                                + (f"- diversity 신규 커버: {', '.join(newly)}\n" if newly else "- diversity: 신규 커버 없음(이미 커버된 축)\n")
                                + ("\n".join(cov_lines) + "\n" if cov_lines else "")
                                + (f"- manufacturer_policy: soft_dedupe(weight={manu_w}) 적용\n" if use_manu else "")
                                + f"- dedupe: key={list(dedupe_key)} allow_axes={list(allow_axes)} allow_duplicates={allow_duplicates} max_dup={max_dup}\n"
                                + f"- tie_breakers: {list(policy.tie_breakers)}"
                            ),
                        }
                    )

        # extras fill:
        # - allow_duplicates: 같은 core에서 max_dup까지 추가 허용
        # - allow_axes: 같은 core라도 axes_key가 다르면 추가 허용 (대표성 이후에만)
        if len(selected_rows) < target_n:
            sub_sorted2 = sub_sorted.copy()
            if used_ids and "device_id" in sub_sorted2.columns:
                sub_sorted2 = sub_sorted2[~sub_sorted2["device_id"].fillna("").astype(str).isin(used_ids)].copy()

            # reuse brand penalty for extras as well (soft)
            if use_manu:
                _brand_diverse_fill(
                    sub_sorted2,
                    selected_rows=selected_rows,
                    target_n=target_n,
                    used_ids=used_ids,
                    core_counts=core_counts,
                    max_dup=max_dup,
                    penalty_weight=manu_w,
                    require_core_col="_profile_core",
                    can_take=(lambda rr: _can_take(rr, is_required=False)),
                    on_take=_mark_taken,
                )
            else:
                for _, rr in sub_sorted2.iterrows():
                    if len(selected_rows) >= target_n:
                        break
                    core = str(rr.get("_profile_core") or "")
                    if core_counts.get(core, 0) >= max_dup:
                        continue
                    # if allow_axes enabled, require new axes_key per core
                    if allow_axes:
                        axk = str(rr.get("_axes_key") or "")
                        if axk in core_axes_seen.get(core, set()):
                            continue
                    did = str(rr.get("device_id") or "")
                    if did and did not in used_ids:
                        if not _can_take(rr, is_required=False):
                            continue
                        covered_before2 = {ax: set(covered.get(ax, set())) for ax in must_cover_axes}
                        selected_rows.append(rr)
                        used_ids.add(did)
                        _mark_taken(rr)
                        core_counts[core] = core_counts.get(core, 0) + 1
                        axk2 = str(rr.get("_axes_key") or "")
                        core_axes_seen.setdefault(core, set()).add(axk2)
                        newly2: list[str] = []
                        for ax in must_cover_axes:
                            v = str(rr.get(ax) or "").strip().lower()
                            if v and v not in covered.setdefault(ax, set()):
                                newly2.append(f"{ax}={v}")
                                covered[ax].add(v)
                        cov_lines2 = []
                        for ax in must_cover_axes:
                            uni = axis_universe.get(ax, [])
                            cov_b = sorted(covered_before2.get(ax, set()))
                            cov_a = sorted(covered.get(ax, set()))
                            remain = [x for x in uni if x not in set(cov_a)]
                            cov_lines2.append(
                                f"- cover({ax}): {len(cov_b)}/{len(uni)} -> {len(cov_a)}/{len(uni)} "
                                f"(added={newly2 if newly2 else []}, remain={remain[:8]}{'...' if len(remain) > 8 else ''})"
                            )
                        yv = rr.get("_year")
                        dv = rr.get("_disp_score")
                        gp = rr.get("_gpu_pri")
                        decision_rows.append(
                            {
                                "rank": rk,
                                "device_id": did,
                                "No": str(rr.get("No") or ""),
                                "product_name": str(rr.get("product_name") or rr.get("model_name") or ""),
                                "brand": str(rr.get("brand") or ""),
                                "cpu_family": str(rr.get("cpu_family") or ""),
                                "profile_core": core,
                                "axes_key": axk2,
                                "stage": "extra",
                                "why": "부족분 충족(extra)로 추가 선정",
                                "why_detail": (
                                    f"[extra] Rank={rk} 목표={target_n}대 대비 부족분을 채우기 위해 추가 선정\n"
                                    f"- rank_candidates(after_filter/excl)={rank_candidates}, unique_profiles={rank_unique_profiles}\n"
                                    f"- selection_progress: selected={len(selected_rows)}/{target_n} (including required)\n"
                                    f"- profile_core={core}\n"
                                    f"- axes_key={axk2} (allow_axes 후보)\n"
                                    f"- cpu_family={str(rr.get('cpu_family') or '')}\n"
                                    f"- tie_keys: year={yv} disp_score={dv} gpu_pri={gp}\n"
                                    + (f"- diversity 신규 커버: {', '.join(newly2)}\n" if newly2 else "- diversity: 신규 커버 없음\n")
                                    + ("\n".join(cov_lines2) + "\n" if cov_lines2 else "")
                                    + (f"- manufacturer_policy: soft_dedupe(weight={manu_w}) 적용\n" if use_manu else "")
                                    + f"- dedupe: key={list(dedupe_key)} allow_axes={list(allow_axes)} allow_duplicates={allow_duplicates} max_dup={max_dup}\n"
                                    + f"- tie_breakers: {list(policy.tie_breakers)}"
                                ),
                            }
                        )

        # truncate to target_n (but never drop required)
        sel_df = pd.DataFrame(selected_rows) if selected_rows else sub_sorted.head(0).copy()
        if not sel_df.empty:
            if req_in_rank and "device_id" in sel_df.columns:
                sel_df["_is_req"] = sel_df["device_id"].fillna("").astype(str).isin(req_in_rank)
                sel_df = sel_df.sort_values(by=["_is_req"], ascending=[False])
            # if required exceeds target_n, keep all required and log
            req_cnt = int(sel_df["_is_req"].sum()) if "_is_req" in sel_df.columns else 0
            if req_cnt > target_n:
                logs.append(f"[warn] rank={rk} required_count({req_cnt}) > target_n({target_n}); keep all required")
                sel_df = sel_df[sel_df["_is_req"]].copy()
            else:
                sel_df = sel_df.head(target_n).copy()

        got_n = int(len(sel_df))
        shortage = max(0, target_n - got_n)
        reason = "ok" if shortage == 0 else "후보 부족"
        shortage_rows.append(
            {
                "rank": rk,
                "target": target_n,
                "selected": got_n,
                "shortage": shortage,
                "reason": reason,
            }
        )

        # clean internal cols
        # keep explainability columns
        if "_profile_core" in sel_df.columns:
            sel_df["profile_core"] = sel_df["_profile_core"]
        if "_axes_key" in sel_df.columns:
            sel_df["axes_key"] = sel_df["_axes_key"]
        sel_df = sel_df.drop(columns=[c for c in sel_df.columns if str(c).startswith("_")], errors="ignore")
        selected_by_rank[rk] = sel_df

    # combine selected
    all_sel = pd.concat([selected_by_rank[rk] for rk in RANK_ORDER if rk in selected_by_rank], ignore_index=True, sort=False)

    # rule summary (3~5 lines as DF)
    rules = [
        f"project={inputs.project} version={inputs.version}",
        f"candidate_filter: usable_only={policy.candidate_filter.availability.usable_only}, rentable_only={policy.candidate_filter.availability.rentable_only}, required_fields={list(policy.candidate_filter.required_fields)}",
        f"dedupe: within_rank key={list(policy.dedupe.within_rank.key)} allow_axes={list(policy.dedupe.within_rank.allow_duplicates_if_differs_by)} allow_duplicates={policy.dedupe.within_rank.allow_duplicates} max_dup={max_dup}",
        f"diversity: must_cover={list(must_cover_axes)}",
        f"tie_breakers: {list(policy.tie_breakers)}",
    ]
    rule_summary = pd.DataFrame([{"rule_summary": x} for x in rules[:5]])

    return PolicyV2Outputs(
        selected=all_sel,
        selected_by_rank=selected_by_rank,
        dedupe_log=pd.DataFrame(dedupe_rows),
        shortage_summary=pd.DataFrame(shortage_rows),
        decision_log=pd.DataFrame(decision_rows),
        rule_summary=rule_summary,
        logs=logs,
    )


