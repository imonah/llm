#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).parent
OUTPUT_JSON = BASE_DIR / "output.json"
PATTERNS_JSON = BASE_DIR / "bibl" / "EMAIL_PATTERNS.json"
EMAILS_JSON = BASE_DIR / "emails.json"
DOMAIN = "domain.com"


def _slug(s: str) -> str:
    """Lowercase and remove non a-z0-9 characters."""
    if s is None:
        return ""
    s = s.strip().lower()
    # Replace any non-alphanumeric characters with nothing
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _load_output(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_patterns(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pats = data.get("patterns", [])
    # Each pattern is {"name": str, "template": str}
    return [p for p in pats if isinstance(p, dict) and "template" in p]


def _render_local(template: str, first: str, last: str) -> str:
    first_clean = first or ""
    last_clean = last or ""
    # Compute helper tokens
    f = first_clean[:1] if first_clean else ""
    l = last_clean[:1] if last_clean else ""
    last_token = (last_clean.split()[-1] if last_clean.strip() else "")

    mapping = {
        "first": _slug(first_clean),
        "last": _slug(last_clean.replace(" ", "")),  # default: compact last by removing spaces
        "f": _slug(f),
        "l": _slug(l),
        "last_token": _slug(last_token),
    }

    local = template
    for key, val in mapping.items():
        local = local.replace("{" + key + "}", val)
    # Collapse any repeated dots that could occur if values are empty
    local = re.sub(r"\.+", ".", local).strip(".")
    return local


def main():
    if not OUTPUT_JSON.exists():
        print(f"Input not found: {OUTPUT_JSON}")
        return
    if not PATTERNS_JSON.exists():
        print(f"Patterns not found: {PATTERNS_JSON}")
        return

    items = _load_output(OUTPUT_JSON)
    patterns = _load_patterns(PATTERNS_JSON)

    results = []
    for rec in items:
        first = rec.get("FirstName", "")
        last = rec.get("LastName", "")
        clean_name = rec.get("CleanName", (first + (" " + last if last else "")).strip())

        out_obj = {"CleanName": clean_name}
        for idx, p in enumerate(patterns, start=1):
            local = _render_local(p.get("template", "{first}.{last}"), first, last)
            email = f"{local}@{DOMAIN}" if local else f"{_slug(first+last)}@{DOMAIN}"
            out_obj[f"Pattern {idx}"] = email
        results.append(out_obj)

    with open(EMAILS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(results)} emails -> {EMAILS_JSON}")


if __name__ == "__main__":
    main()
