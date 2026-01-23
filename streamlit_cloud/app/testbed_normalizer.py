from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


def _norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    return s


def _first_present(cols_norm_to_raw: dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        k = _norm_col(c)
        raw = cols_norm_to_raw.get(k)
        if raw:
            return raw
    return None


def parse_ram_gb(v: Any) -> Optional[float]:
    """
    Examples:
    - "1.5GB", "3GB", "6 gb", "8" -> float(GB)
    - None/"" -> None
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s2 = s.replace(",", "").strip().lower()

    # 4096MB 같은 케이스까지 일단 커버
    m_mb = re.search(r"(\d+(?:\.\d+)?)\s*mb", s2)
    if m_mb:
        try:
            return float(m_mb.group(1)) / 1024.0
        except Exception:
            return None

    m_gb = re.search(r"(\d+(?:\.\d+)?)\s*gb", s2)
    if m_gb:
        try:
            return float(m_gb.group(1))
        except Exception:
            return None

    # 숫자만 있는 경우
    m_num = re.search(r"(\d+(?:\.\d+)?)", s2)
    if m_num:
        try:
            return float(m_num.group(1))
        except Exception:
            return None

    return None


def ram_tier_from_gb(ram_gb: Optional[float]) -> Optional[str]:
    if ram_gb is None:
        return None
    if ram_gb <= 3.0:
        return "Low"
    if 4.0 <= ram_gb <= 6.0:
        return "Mid"
    if ram_gb >= 8.0:
        return "High"
    # 7GB 등 애매한 값은 Mid로 묶지 않고 None 처리(원본 정합성 확인용)
    return None


def parse_display_resolution(v: Any) -> tuple[Optional[int], Optional[int]]:
    """
    Examples:
    - "1560 x 720", "720x1560", "1080*2400" -> (w, h)
    Rule: 입력이 "1560 x 720"처럼 뒤바뀌어도 숫자 2개를 잡아서 그대로 저장.
    (추후 표준화 필요 시 여기서 min/max 기준으로 바꿀 수 있음)
    """
    if v is None:
        return (None, None)
    s = str(v).strip().lower()
    if not s:
        return (None, None)
    s = s.replace("×", "x").replace("*", "x")
    m = re.search(r"(\d{3,5})\s*x\s*(\d{3,5})", s)
    if not m:
        return (None, None)
    try:
        a = int(m.group(1))
        b = int(m.group(2))
        return (a, b)
    except Exception:
        return (None, None)


def res_class_from_resolution(w: Optional[int], h: Optional[int]) -> Optional[str]:
    """
    v1 간단 규칙 기반 분류(조정 가능):
    - low_res: short_edge < 720
    - standard: short_edge == 720
    - fhd: short_edge == 1080
    - high_res: short_edge >= 1440
    - unknown: 그 외
    """
    # pandas가 None을 NaN(float)로 바꿔 전달할 수 있어 보정
    try:
        if w is None or h is None:
            return None
        if isinstance(w, float) and pd.isna(w):  # type: ignore[name-defined]
            return None
        if isinstance(h, float) and pd.isna(h):  # type: ignore[name-defined]
            return None
    except Exception:
        return None

    if not w or not h:
        return None
    short_edge = min(int(w), int(h))
    if short_edge < 720:
        return "low_res"
    if short_edge == 720:
        return "standard"
    if short_edge == 1080:
        return "tall_fhd"
    if short_edge >= 1440:
        return "high_res"
    return "unknown"


def parse_boolish(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    # Excel에서 1/0이 float(1.0/0.0)로 들어오는 경우 처리
    try:
        if isinstance(v, (int, float)) and not pd.isna(v):
            return bool(int(v))
    except Exception:
        pass
    s = str(v).strip().lower()
    if not s:
        return None
    # "1.0"/"0.0" 같은 케이스
    try:
        if re.fullmatch(r"\d+(?:\.\d+)?", s):
            return bool(int(float(s)))
    except Exception:
        pass
    if s in ("1", "y", "yes", "true", "가능", "대여가능", "ok"):
        return True
    if s in ("0", "n", "no", "false", "불가", "대여불가", "x"):
        return False
    return None


def is_broken_status(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    if not s:
        return False
    # "고장", "폐기", "분실" 등은 사용 불가로 처리
    return any(tok in s for tok in ("고장", "폐기", "분실", "파손"))


@dataclass(frozen=True)
class TestbedSchema:
    # 표준 컬럼
    device_id: str = "device_id"
    # 원본이 '제품명', '모델번호'로 분리되어 있으므로 둘 다 표준 컬럼으로 보존한다.
    product_name: str = "product_name"
    model_number: str = "model_number"
    # 하위 호환(기존 사용처가 model_name을 기대할 수 있음): product_name과 동일하게 둔다.
    model_name: str = "model_name"
    brand: str = "brand"
    ap_family: str = "ap_family"
    gpu: str = "gpu"
    ram_gb: str = "ram_gb"
    ram_tier: str = "ram_tier"
    resolution_w: str = "resolution_w"
    resolution_h: str = "resolution_h"
    res_class: str = "res_class"
    available: str = "available"


def load_testbed(path: Path, sheet: Optional[str] = None, header_row_1based: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        # pandas: sheet_name=None -> dict(sheet_name->DataFrame).
        # 기본 동작: 첫 번째 시트.
        # 하지만 실제 마스터 테이블이 Device_Info 시트에 있는 경우가 많아서, sheet 미지정이면 Device_Info를 우선 사용.
        sheet_name: object
        if sheet is None:
            try:
                xls = pd.ExcelFile(path)
                preferred = None
                for sname in list(xls.sheet_names):
                    if str(sname).strip().lower() == "device_info":
                        preferred = sname
                        break
                sheet_name = preferred if preferred is not None else 0
            except Exception:
                sheet_name = 0
        else:
            sheet_name = sheet

        # openpyxl이 일부 확장 요소에 대해 UserWarning을 띄울 수 있어(기능에는 영향 없음) 로그만 정리한다.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Unknown extension is not supported and will be removed",
                category=UserWarning,
            )

            # 1) 헤더 행이 고정(예: 6행)인 경우: 명시적으로 해당 행을 header로 사용
            if header_row_1based is not None:
                header_idx = max(0, int(header_row_1based) - 1)
                data = pd.read_excel(path, sheet_name=sheet_name, header=header_idx)  # type: ignore[call-arg]
            else:
                # 2) 그 외: 헤더 위치 자동 탐지(상단 N줄에서 기대 토큰 매칭이 가장 많은 행)
                preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=30)  # type: ignore[call-arg]
                if isinstance(preview, dict):
                    if not preview:
                        return pd.DataFrame()
                    preview = preview[next(iter(preview.keys()))]

                expected = {
                    "no",
                    "자산번호",
                    "제품명",
                    "모델번호",
                    "제조사",
                    "cpu",
                    "gpu",
                    "ram",
                    "display",
                    "상태",
                    "대여가능여부",
                }

                best_idx = 0
                best_score = -1
                for i in range(min(len(preview), 30)):
                    row = preview.iloc[i].tolist()
                    tokens = {
                        str(x).strip().lower()
                        for x in row
                        if x is not None and str(x).strip() not in ("", "nan")
                    }
                    score = sum(1 for t in tokens if t in expected)
                    if score > best_score:
                        best_score = score
                        best_idx = i

                data = pd.read_excel(path, sheet_name=sheet_name, header=best_idx)  # type: ignore[call-arg]

            if isinstance(data, dict):
                if not data:
                    return pd.DataFrame()
                data = data[next(iter(data.keys()))]
            # 완전 빈 컬럼 제거(엑셀 빈 영역이 컬럼으로 잡히는 경우)
            data = data.dropna(axis=1, how="all")
            return data
    if suffix in (".csv",):
        return pd.read_csv(path)
    raise ValueError(f"지원하지 않는 파일 형식: {path.name}")


def normalize_testbed(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    원본 컬럼명이 프로젝트마다 흔들려도, 내부 표준 스키마로 정규화한다.
    """
    schema = TestbedSchema()
    df = df_raw.copy()

    cols_norm_to_raw = {_norm_col(c): c for c in df.columns}

    # 원본 컬럼 매핑(유연)
    c_device_id = _first_present(cols_norm_to_raw, ["자산번호", "device_id", "asset_id", "id"])
    c_product = _first_present(cols_norm_to_raw, ["제품명", "product_name", "model_name", "model", "device", "기기명"])
    c_modelno = _first_present(cols_norm_to_raw, ["모델번호", "model_number", "modelno", "model_no", "sku"])
    c_brand = _first_present(cols_norm_to_raw, ["제조사", "brand", "maker", "vendor"])
    c_cpu = _first_present(cols_norm_to_raw, ["cpu", "ap", "soc", "칩셋", "ap_family"])
    c_gpu = _first_present(cols_norm_to_raw, ["gpu", "그래픽", "gpu_model"])
    c_ram = _first_present(cols_norm_to_raw, ["ram", "ram_gb", "memory", "메모리"])
    c_disp = _first_present(cols_norm_to_raw, ["display", "해상도", "resolution", "screen", "디스플레이"])
    c_rent = _first_present(cols_norm_to_raw, ["대여가능여부", "대여가능", "rentable", "available"])
    c_status = _first_present(cols_norm_to_raw, ["상태", "status", "condition"])

    def _get_series(col: Optional[str]) -> pd.Series:
        if not col:
            return pd.Series([None] * len(df))
        return df[col]

    s_device_id = _get_series(c_device_id).astype(str).str.strip().replace({"nan": ""})
    s_product = _get_series(c_product).astype(str).str.strip().replace({"nan": ""})
    s_modelno = _get_series(c_modelno).astype(str).str.strip().replace({"nan": ""})
    s_brand = _get_series(c_brand).astype(str).str.strip().replace({"nan": ""})
    s_cpu = _get_series(c_cpu).astype(str).str.strip().replace({"nan": ""})
    s_gpu = _get_series(c_gpu).astype(str).str.strip().replace({"nan": ""})
    s_ram = _get_series(c_ram)
    s_disp = _get_series(c_disp)
    s_rent = _get_series(c_rent)
    s_status = _get_series(c_status)

    ram_gb = s_ram.apply(parse_ram_gb)
    ram_tier = ram_gb.apply(ram_tier_from_gb)

    res_pairs = s_disp.apply(parse_display_resolution)
    resolution_w = res_pairs.apply(lambda t: t[0])
    resolution_h = res_pairs.apply(lambda t: t[1])
    res_class = pd.DataFrame({"w": resolution_w, "h": resolution_h}).apply(
        lambda r: res_class_from_resolution(r["w"], r["h"]), axis=1
    )

    rent_bool = s_rent.apply(parse_boolish)
    broken_bool = s_status.apply(is_broken_status)
    # FutureWarning(다운캐스팅) 방지: boolean dtype로 정규화 후 연산
    rent_ok = rent_bool.astype("boolean").fillna(False)
    broken = broken_bool.astype(bool)

    # available 판정(v2 POC 요구):
    # - "상태"가 사용가능 또는 CQA 인 경우에만 후보로 포함
    # - 고장/폐기/분실 등 broken 상태는 항상 제외
    if c_status:
        status_txt = s_status.fillna("").astype(str).str.strip()
        status_norm = status_txt.str.replace(" ", "").str.lower()
        status_ok = status_norm.str.contains("사용가능", na=False) | status_norm.str.contains("cqa", na=False)
        available = status_ok.astype(bool) & (~broken)
    else:
        # 상태 컬럼이 없으면 기존 로직으로 fallback
        available = rent_ok.astype(bool) & (~broken)

    std = pd.DataFrame(
        {
            schema.device_id: s_device_id.replace({"": None}),
            schema.product_name: s_product.replace({"": None}),
            schema.model_number: s_modelno.replace({"": None}),
            schema.model_name: s_product.replace({"": None}),  # alias
            schema.brand: s_brand.replace({"": None}),
            schema.ap_family: s_cpu.replace({"": None}),
            schema.gpu: s_gpu.replace({"": None}),
            schema.ram_gb: ram_gb,
            schema.ram_tier: ram_tier,
            schema.resolution_w: resolution_w,
            schema.resolution_h: resolution_h,
            schema.res_class: res_class,
            schema.available: available,
        }
    )

    # 출력 컬럼 순서(v1 요구):
    # - 엑셀 "원본 헤더 순서"를 그대로 유지
    # - 표준(정규화) 컬럼은 해당 원본 헤더 "바로 옆"에 끼워 넣는다.
    raw = df_raw.copy()
    rename_map: dict[str, str] = {}
    for c in raw.columns:
        if c in std.columns:
            rename_map[c] = f"raw__{c}"
    if rename_map:
        raw = raw.rename(columns=rename_map)

    # 어떤 원본 컬럼 옆에 어떤 표준 컬럼을 넣을지 정의
    insert_after: dict[str, list[str]] = {}
    if c_device_id:
        insert_after.setdefault(c_device_id, []).append(schema.device_id)
    if c_product:
        insert_after.setdefault(c_product, []).extend([schema.product_name, schema.model_name])
    if c_modelno:
        insert_after.setdefault(c_modelno, []).append(schema.model_number)
    if c_brand:
        insert_after.setdefault(c_brand, []).append(schema.brand)
    if c_cpu:
        insert_after.setdefault(c_cpu, []).append(schema.ap_family)
    if c_gpu:
        insert_after.setdefault(c_gpu, []).append(schema.gpu)
    if c_ram:
        insert_after.setdefault(c_ram, []).extend([schema.ram_gb, schema.ram_tier])
    if c_disp:
        insert_after.setdefault(c_disp, []).extend([schema.resolution_w, schema.resolution_h, schema.res_class])
    # available은 rent/status가 모두 있으면 status 뒤에, 없으면 rent 뒤에, 둘 다 없으면 마지막
    if c_status:
        insert_after.setdefault(c_status, []).append(schema.available)
    elif c_rent:
        insert_after.setdefault(c_rent, []).append(schema.available)

    std_cols_added: set[str] = set()
    ordered_cols: list[str] = []

    # 원본 컬럼들(엑셀 헤더 순서 유지) + 옆에 표준 컬럼 삽입
    for c in raw.columns:
        ordered_cols.append(c)
        for sc in insert_after.get(c, []):
            if sc in std.columns and sc not in std_cols_added:
                ordered_cols.append(sc)
                std_cols_added.add(sc)

    # anchor가 없어서 삽입되지 않은 표준 컬럼은 마지막에 추가(빠짐 방지)
    for sc in std.columns:
        if sc not in std_cols_added:
            ordered_cols.append(sc)
            std_cols_added.add(sc)

    out = pd.concat([raw.reset_index(drop=True), std], axis=1)[ordered_cols]

    # 최소 필수값이 없는 행은 제거(자산번호/제품명 둘 다 없으면 노이즈로 간주)
    out = out[~(out[schema.device_id].isna() & out[schema.product_name].isna())].reset_index(drop=True)
    return out


