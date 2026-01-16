from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DotEnvLoadResult:
    path: Path
    loaded_keys: list[str]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _parse_env_value(raw: str) -> str:
    """
    dotenv 스타일 값 파싱:
    - KEY="value"  # comment   형태를 지원 (따옴표 뒤의 주석은 무시)
    - KEY=value # comment      형태를 지원 (공백 이후 #부터 주석)
    - 복잡한 escape는 다루지 않고, 실사용에 필요한 범위만 처리
    """
    s = raw.strip()
    if not s:
        return ""

    # quoted value: take content inside the first matching quote, ignore the rest (comments)
    if s[0] in ("'", '"'):
        q = s[0]
        end = s.find(q, 1)
        if end != -1:
            return s[1:end]
        # fallback (unterminated quote)
        return _strip_quotes(s)

    # unquoted: strip inline comment that starts with # after whitespace
    cut = len(s)
    for i, ch in enumerate(s):
        if ch == "#":
            # if at beginning or preceded by whitespace => comment
            if i == 0 or s[i - 1].isspace():
                cut = i
                break
    return s[:cut].strip()


def load_dotenv(path: Path, *, override: bool = False) -> Optional[DotEnvLoadResult]:
    """
    Minimal .env loader (no external deps).
    - KEY=VALUE
    - ignores empty lines / comments (# ...)
    - does NOT override existing env by default
    """
    if not path.exists():
        return None

    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # PowerShell style: $env:KEY = "VALUE"
        if line.lower().startswith("$env:") and "=" in line:
            left, right = line.split("=", 1)
            # "$env:KEY" or "$env:KEY   "
            key = left.strip()[5:].strip()  # remove "$env:"
            if not key:
                continue
            val = _parse_env_value(right)
            if (not override) and (key in os.environ) and os.environ.get(key):
                continue
            os.environ[key] = val
            loaded.append(key)
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            continue
        val = _parse_env_value(v)
        if (not override) and (key in os.environ) and os.environ.get(key):
            continue
        os.environ[key] = val
        loaded.append(key)

    return DotEnvLoadResult(path=path, loaded_keys=loaded)


def load_default_env(project_root: Path, *, override: bool = False) -> Optional[DotEnvLoadResult]:
    """
    devices_auto에서 기본으로 읽을 env 파일:
    1) local.env (로컬 전용; dotfile 생성이 막힌 환경에서도 사용 가능)
    2) .env.local
    3) .env
    4) env.example (사용자가 임시로 여기에 넣는 경우가 있어 fallback)
    """
    res = load_dotenv(project_root / "local.env", override=override)
    if res is not None:
        return res
    res = load_dotenv(project_root / ".env.local", override=override)
    if res is not None:
        return res
    res = load_dotenv(project_root / ".env", override=override)
    if res is not None:
        return res
    return load_dotenv(project_root / "env.example", override=override)


def find_project_root(start: Path) -> Path:
    """
    devices_auto 프로젝트 루트(= scripts/, app/가 있는 폴더)를 찾는다.
    start가 scripts/ 또는 app/ 내부여도 상위로 올라가며 탐색한다.
    """
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(6):
        if (cur / "app").is_dir() and (cur / "scripts").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parent if start.is_file() else start.resolve()


