from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
import re


@dataclass(frozen=True)
class AvailabilityFilter:
    usable_only: bool = True
    rentable_only: bool = False


@dataclass(frozen=True)
class CandidateFilter:
    platform: str = "android"
    required_fields: tuple[str, ...] = ()
    # numeric ranges filter (inclusive): e.g. {ram_gb: [2, 3]}
    # values are stored as dict[col] = (min, max)
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    availability: AvailabilityFilter = AvailabilityFilter()


@dataclass(frozen=True)
class DedupeWithinRank:
    key: tuple[str, ...] = ("cpu", "gpu", "ram_gb")
    allow_duplicates: bool = False
    max_duplicates_per_profile: int = 1
    allow_duplicates_if_differs_by: tuple[str, ...] = ()
    # additional constraint: limit how many devices with same product_name can be selected within a rank
    # 0 means "off"
    max_per_product_name: int = 0


@dataclass(frozen=True)
class DedupeConfig:
    within_rank: DedupeWithinRank = DedupeWithinRank()


@dataclass(frozen=True)
class DiversityWithinRank:
    must_cover: tuple[str, ...] = ()
    optional_axes: tuple[str, ...] = ()
    # optional hard inclusion if available within the rank:
    # e.g. {form_factor: [pad]} => if any candidate has form_factor=pad and target_n>0, ensure at least one pad is selected
    must_include_if_available: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class DiversityConfig:
    within_rank: DiversityWithinRank = DiversityWithinRank()


@dataclass(frozen=True)
class MemoryWindowConfig:
    avoid_previous_version_models: bool = False
    allow_n_minus_2_models: bool = True


@dataclass(frozen=True)
class VersioningConfig:
    memory_window: MemoryWindowConfig = MemoryWindowConfig()


@dataclass(frozen=True)
class FocusConfig:
    preferred_ranks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManufacturerWithinRank:
    """
    제조사(brand) 정책:
    - mode:
      - "off": 사용 안 함
      - "soft_dedupe": CPU/GPU/RAM 대표성(=프로파일) 확보 이후, 동일 제조사 중복을 완화된 방식으로 뒤로 민다
    - penalty_weight: 동일 제조사 재사용 패널티 가중치(0~1 권장)
    """

    mode: str = "off"
    penalty_weight: float = 0.0


@dataclass(frozen=True)
class ManufacturerPolicy:
    within_rank: ManufacturerWithinRank = ManufacturerWithinRank()


@dataclass(frozen=True)
class PolicyV2:
    project: str
    version: str = "v2"
    candidate_filter: CandidateFilter = CandidateFilter()
    dedupe: DedupeConfig = DedupeConfig()
    diversity: DiversityConfig = DiversityConfig()
    tie_breakers: tuple[object, ...] = ()
    versioning: Optional[VersioningConfig] = None
    focus: Optional[FocusConfig] = None
    manufacturer_policy: Optional[ManufacturerPolicy] = None


def _get(d: dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_policy_v2(path: str | Path) -> PolicyV2:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("policy_v2 파싱 실패: 최상위가 dict가 아닙니다.")

    project = str(obj.get("project") or "").strip()
    if not project:
        raise RuntimeError("policy_v2: project가 비어 있습니다.")

    # candidate_filter
    av = AvailabilityFilter(
        usable_only=bool(_get(obj, "candidate_filter.availability.usable_only", True)),
        rentable_only=bool(_get(obj, "candidate_filter.availability.rentable_only", False)),
    )
    # numeric_ranges (optional)
    nr_obj = _get(obj, "candidate_filter.numeric_ranges", {}) or {}
    numeric_ranges: dict[str, tuple[float, float]] = {}
    if isinstance(nr_obj, dict):
        for k, v in nr_obj.items():
            if not k:
                continue
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    lo = float(v[0])
                    hi = float(v[1])
                    numeric_ranges[str(k)] = (lo, hi)
                except Exception:
                    continue

    cf = CandidateFilter(
        platform=str(_get(obj, "candidate_filter.platform", "android") or "android"),
        required_fields=tuple(str(x) for x in (_get(obj, "candidate_filter.required_fields", []) or [])),
        numeric_ranges=numeric_ranges,
        availability=av,
    )

    # dedupe
    dwr = DedupeWithinRank(
        key=tuple(str(x) for x in (_get(obj, "dedupe.within_rank.key", ["cpu", "gpu", "ram_gb"]) or [])),
        allow_duplicates=bool(_get(obj, "dedupe.within_rank.allow_duplicates", False)),
        max_duplicates_per_profile=int(_get(obj, "dedupe.within_rank.max_duplicates_per_profile", 1) or 1),
        allow_duplicates_if_differs_by=tuple(
            str(x) for x in (_get(obj, "dedupe.within_rank.allow_duplicates_if_differs_by", []) or [])
        ),
        max_per_product_name=int(_get(obj, "dedupe.within_rank.max_per_product_name", 0) or 0),
    )
    dedupe = DedupeConfig(within_rank=dwr)

    # diversity
    mia_obj = _get(obj, "diversity.within_rank.must_include_if_available", {}) or {}
    mia: dict[str, tuple[str, ...]] = {}
    if isinstance(mia_obj, dict):
        for k, v in mia_obj.items():
            if not k:
                continue
            if isinstance(v, (list, tuple)):
                mia[str(k)] = tuple(str(x).strip().lower() for x in v if str(x).strip())
    dvr = DiversityWithinRank(
        must_cover=tuple(str(x) for x in (_get(obj, "diversity.within_rank.must_cover", []) or [])),
        optional_axes=tuple(str(x) for x in (_get(obj, "diversity.within_rank.optional_axes", []) or [])),
        must_include_if_available=mia,
    )
    diversity = DiversityConfig(within_rank=dvr)

    # tie_breakers (kept as raw list to preserve extension)
    tbs = tuple(obj.get("tie_breakers") if isinstance(obj.get("tie_breakers"), list) else [])

    # versioning (optional)
    versioning = None
    if isinstance(obj.get("versioning"), dict):
        mw = MemoryWindowConfig(
            avoid_previous_version_models=bool(_get(obj, "versioning.memory_window.avoid_previous_version_models", False)),
            allow_n_minus_2_models=bool(_get(obj, "versioning.memory_window.allow_n_minus_2_models", True)),
        )
        versioning = VersioningConfig(memory_window=mw)

    # focus (optional)
    focus = None
    if isinstance(obj.get("focus"), dict):
        focus = FocusConfig(preferred_ranks=tuple(str(x) for x in (_get(obj, "focus.preferred_ranks", []) or [])))

    # manufacturer_policy (optional)
    manufacturer_policy = None
    if isinstance(obj.get("manufacturer_policy"), dict):
        mode = str(_get(obj, "manufacturer_policy.within_rank.mode", "off") or "off").strip()
        try:
            w = float(_get(obj, "manufacturer_policy.within_rank.penalty_weight", 0.0) or 0.0)
        except Exception:
            w = 0.0
        manufacturer_policy = ManufacturerPolicy(within_rank=ManufacturerWithinRank(mode=mode, penalty_weight=w))

    return PolicyV2(
        project=project,
        version=str(obj.get("version") or "v2"),
        candidate_filter=cf,
        dedupe=dedupe,
        diversity=diversity,
        tie_breakers=tbs,
        versioning=versioning,
        focus=focus,
        manufacturer_policy=manufacturer_policy,
    )


def normalize_df_for_policy_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    현재 코드베이스의 testbed_normalized/eligible_candidates 컬럼을 v2 표준 필드로 매핑한다.
    - rank: Rank/Rating/raw__Rank/raw__Rating 등을 우선 사용 (없으면 빈 값)
    - cpu: ap_family
    - gpu: gpu 또는 GPU
    - display_w/h: resolution_w/resolution_h
    """
    out = df.copy()

    def _first_col(cands: list[str]) -> Optional[str]:
        for c in cands:
            if c in out.columns:
                return c
        return None

    def _norm_key(x: object) -> str:
        # remove ALL whitespace (including NBSP) + common punctuations + BOM
        s = str(x or "").replace("\ufeff", "").strip().lower()
        s = re.sub(r"\s+", "", s)
        for ch in ["-", "_", ".", "(", ")", "[", "]", "{", "}", ":", ";", "/", "\\", "|"]:
            s = s.replace(ch, "")
        return s

    # rank
    rank_col = _first_col(["Rank", "rank", "Rating", "raw__Rank", "raw__Rating"])
    if rank_col:
        r = out[rank_col].fillna("").astype(str).str.strip().str.upper().str.replace(" ", "")
    else:
        r = pd.Series([""] * len(out), index=out.index)
    out["rank"] = r

    # No: normalize common variations into a canonical "No" column
    # - 같은 norm key("no")를 가진 컬럼이 여러 개일 수 있어(빈 No + 실제 No)
    #   사진 기준(AP/IP/AT/IT/R로 시작)을 만족하는 값을 가장 많이 포함하는 컬럼을 우선 선택한다.
    NO_CODE_RE = re.compile(r"(?i)^(AP|IP|AT|IT|R)\\d+")
    candidates = ["No", "NO", "no", "No.", "NO.", "번호", "관리번호", "자산No", "자산 No", "AssetNo", "Asset No", "raw__No", "raw__NO"]
    norm_to_cols: dict[str, list[str]] = {}
    for c in out.columns:
        k = _norm_key(c)
        if not k:
            continue
        norm_to_cols.setdefault(k, []).append(str(c))

    def _best_non_empty(cols: list[str]) -> str | None:
        best: str | None = None
        best_score = (-1, -1)  # (pattern_match_count, non_empty_count)
        for c in cols:
            if c not in out.columns:
                continue
            try:
                s = out[c].fillna("").astype(str).str.strip()
                non_empty = int(s.ne("").sum())
                pat = int(s.map(lambda x: bool(NO_CODE_RE.match(str(x)))).sum())
            except Exception:
                non_empty = 0
                pat = 0
            score = (pat, non_empty)
            if score > best_score:
                best_score = score
                best = c
        return best if best_score[1] > 0 else (best or None)

    picked: str | None = None
    for cand in candidates:
        k = _norm_key(cand)
        cols = norm_to_cols.get(k) or []
        best = _best_non_empty(cols) if cols else None
        if best:
            picked = best
            break

    # fallback: if any column normalizes exactly to "no", pick best among them
    if not picked:
        cols = norm_to_cols.get("no") or []
        picked = _best_non_empty(cols) if cols else None

    if picked:
        out["No"] = out[picked].fillna("").astype(str).str.strip()
    else:
        out["No"] = out.get("No", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.strip()

    # platform: derive from OS text (best-effort)
    os_col = _first_col(["OS", "os", "raw__OS", "OS Ver", "OSVer", "os_ver", "OS Version", "OS_VERSION"])
    if os_col:
        os_s = out[os_col].fillna("").astype(str).str.strip().str.lower()
        ios_m = os_s.str.contains("ios") | os_s.str.contains("iphone") | os_s.str.contains("ipad")
        out["platform"] = ios_m.map(lambda x: "ios" if bool(x) else "android")
    elif "platform" in out.columns:
        out["platform"] = out["platform"].fillna("").astype(str).str.strip().str.lower().replace({"aos": "android"})
    else:
        out["platform"] = "android"

    # cpu/gpu/ram/display
    cpu_src = _first_col(["ap_family", "cpu", "CPU", "AP/SoC", "AP", "SoC"])
    if cpu_src:
        out["cpu"] = out[cpu_src].fillna("").astype(str).str.strip()
    else:
        out["cpu"] = ""

    # cpu_family (chipset family) derived from cpu string
    def _cpu_family(x: object) -> str:
        s = str(x or "").strip().lower()
        if not s:
            return "unknown"
        if "snapdragon" in s or "qualcomm" in s:
            return "snapdragon"
        if "exynos" in s:
            return "exynos"
        if "dimensity" in s or "mediatek" in s:
            return "dimensity"
        if "kirin" in s:
            return "kirin"
        if "tensor" in s:
            return "tensor"
        if "helio" in s:
            return "helio"
        return "other"

    out["cpu_family"] = out["cpu"].map(_cpu_family)

    gpu_col = _first_col(["gpu", "GPU"])
    if gpu_col:
        out["gpu"] = out[gpu_col].fillna("").astype(str).str.strip()
    else:
        out["gpu"] = ""

    # ram_gb: parse from common RAM columns if needed
    if "ram_gb" in out.columns:
        out["ram_gb"] = pd.to_numeric(out["ram_gb"], errors="coerce")
    else:
        ram_src = _first_col(["RAM(GB)", "RAM", "ram", "_ram"])
        if ram_src:
            sram = out[ram_src].fillna("").astype(str).str.strip()
            # extract number like "16GB", "16", "16 gb"
            out["ram_gb"] = pd.to_numeric(sram.str.extract(r"(\d+(\.\d+)?)")[0], errors="coerce")
        else:
            out["ram_gb"] = pd.Series([pd.NA] * len(out), index=out.index)

    # display_w/h: from resolution_w/h or parse from DISPLAY-like column
    if "resolution_w" in out.columns:
        out["display_w"] = pd.to_numeric(out["resolution_w"], errors="coerce")
    if "resolution_h" in out.columns:
        out["display_h"] = pd.to_numeric(out["resolution_h"], errors="coerce")
    if "display_w" not in out.columns or "display_h" not in out.columns:
        disp_src = _first_col(["DISPLAY", "display", "해상도", "Resolution", "resolution"])
        if disp_src:
            sdisp = out[disp_src].fillna("").astype(str).str.lower()
            # accept "2354 x 1080" / "2354*1080" / "2354×1080"
            m = sdisp.str.replace("×", "x").str.replace("*", "x").str.extract(r"(\d+)\s*x\s*(\d+)")
            if "display_w" not in out.columns:
                out["display_w"] = pd.to_numeric(m[0], errors="coerce")
            if "display_h" not in out.columns:
                out["display_h"] = pd.to_numeric(m[1], errors="coerce")
        else:
            if "display_w" not in out.columns:
                out["display_w"] = pd.Series([pd.NA] * len(out), index=out.index)
            if "display_h" not in out.columns:
                out["display_h"] = pd.Series([pd.NA] * len(out), index=out.index)

    # display bucket: prefer res_class if present, else derive from short edge
    if "res_class" in out.columns:
        out["display_bucket"] = out["res_class"].fillna("").astype(str).str.strip().str.lower()
    else:
        try:
            w = pd.to_numeric(out.get("display_w", pd.Series([None] * len(out), index=out.index)), errors="coerce")
            h = pd.to_numeric(out.get("display_h", pd.Series([None] * len(out), index=out.index)), errors="coerce")
            short = pd.concat([w, h], axis=1).min(axis=1)
            out["display_bucket"] = (
                short.apply(lambda x: "unknown" if pd.isna(x) else ("standard" if int(x) == 720 else ("tall_fhd" if int(x) == 1080 else ("high_res" if int(x) >= 1440 else "low_res"))))
                .astype(str)
                .str.lower()
            )
        except Exception:
            out["display_bucket"] = ""

    # form_factor / target_market: best-effort from common columns if present
    if "디바이스 타입" in out.columns and "form_factor" not in out.columns:
        out["form_factor"] = out["디바이스 타입"].fillna("").astype(str).str.strip().str.lower()
    elif "form_factor" not in out.columns:
        out["form_factor"] = ""

    if "타겟 국가" in out.columns and "target_market" not in out.columns:
        out["target_market"] = out["타겟 국가"].fillna("").astype(str).str.strip().str.lower()
    elif "target_market" not in out.columns:
        out["target_market"] = ""

    # note: best-effort from common columns (for exclusion dropdown)
    note_col = _first_col(["note", "NOTE", "Note", "비고", "raw__NOTE", "raw__note", "raw__비고"])
    if note_col:
        out["note"] = out[note_col].fillna("").astype(str).str.strip()
    elif "note" not in out.columns:
        out["note"] = ""

    # release_year: best-effort parse from common columns
    year_col = _first_col(["출시 년도", "출시년도", "release_year", "year"])
    if year_col:
        s = out[year_col].fillna("").astype(str)
        # extract 4-digit year
        out["release_year"] = pd.to_numeric(s.str.extract(r"(\d{4})")[0], errors="coerce")
    else:
        out["release_year"] = pd.Series([pd.NA] * len(out), index=out.index)

    # product/brand/device_id
    if "product_name" in out.columns:
        out["product_name"] = out["product_name"].fillna("").astype(str).str.strip()
    if "brand" in out.columns:
        out["brand"] = out["brand"].fillna("").astype(str).str.strip()
    if "device_id" in out.columns:
        out["device_id"] = out["device_id"].fillna("").astype(str).str.strip()

    return out


