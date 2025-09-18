#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Tuple, Set, Optional

# ---------- НОРМАЛИЗАЦИЯ ТЕКСТА ----------

_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    """
    1) заменяет запятые/точки на пробел (чтобы убирать 'Dr.' и '..., PhD')
    2) схлопывает все пробелы (включая нестандартные юникод-пробелы)
    3) обрезает пробелы по краям
    """
    if s is None:
        return ""
    # Удаляем всё содержимое в любых видах скобок (включая сами скобки)
    s = _strip_bracketed(s)
    s = re.sub(r"[.,]", " ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()

def strip_name_punctuation(token: str) -> str:
    """Убирает точку на концах ('Dr.' -> 'Dr')."""
    return token.strip().strip(".,").strip()

# ---------- ЗАГРУЗКА СЛОВАРЕЙ ----------

def _strip_bracketed(text: str) -> str:
    """Удаляет подстроки в скобках любых типов: (), [], {}, <>, а также их распространённые юникод-аналоги.
    Повторяет проход, чтобы убрать вложенные случаи.
    """
    if not text:
        return text
    # Набор пар скобок: обычные ASCII и распространённые юникод-варианты
    patterns = [
        ("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"),
        ("（", "）"), ("［", "］"), ("｛", "｝"), ("〈", "〉"), ("《", "》"), ("＜", "＞")
    ]
    # Строим regex для каждой пары: убираем неперекрывающиеся фрагменты без вложенности, повторяем до стабилизации
    changed = True
    while changed:
        changed = False
        start = text
        for l, r in patterns:
            # Экранируем скобки для regex
            lr = re.escape(l)
            rr = re.escape(r)
            text = re.sub(fr"{lr}[^{lr}{rr}]*{rr}", " ", text)
        if text != start:
            changed = True
    return text

def _normalize_vocab(values: Iterable[str]) -> Set[str]:
    """Нормализует словарные значения: нижний регистр, без точек и лишних пробелов."""
    out = set()
    for v in values or []:
        if not v:
            continue
        v = strip_name_punctuation(str(v)).lower()
        if v:
            out.add(v)
    return out

def _normalize_particles(values: Iterable[str]) -> Set[str]:
    """Нормализует частицы фамилии: нижний регистр, убирает точки по краям."""
    out = set()
    for v in values or []:
        if not v:
            continue
        v = strip_name_punctuation(str(v)).lower()
        if v:
            out.add(v)
    return out

def _load_list(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Поддерживаем форматы: ["a","b"] или {"values":[...]} на будущее
    if isinstance(data, dict):
        return data.get("values", [])
    return data

def load_vocab() -> Tuple[Set[str], Set[str]]:
    """
    Загружает словари титулов/постфиксов из файлов в папке bibl/:
      - bibl/DEFAULT_TITLES.json
      - bibl/DEFAULT_SUFFIXES.json
    Возвращает множества с нормализованными значениями.
    """
    base = Path(__file__).parent / "bibl"
    titles_list = _load_list(base / "DEFAULT_TITLES.json")
    suffixes_list = _load_list(base / "DEFAULT_SUFFIXES.json")
    titles = _normalize_vocab(titles_list)
    suffixes = _normalize_vocab(suffixes_list)
    return titles, suffixes

def load_lastname_particles() -> Set[str]:
    """
    Загружает частицы фамилий из файла в папке bibl/:
      - bibl/LASTNAME_PARTICLES.json
    Возвращает множество с нормализованными значениями.
    """
    base = Path(__file__).parent / "bibl"
    particles_list = _load_list(base / "LASTNAME_PARTICLES.json")
    return _normalize_particles(particles_list)

# ---------- ВСПОМОГАТЕЛЬНЫЕ НАБОРЫ ----------

LASTNAME_PARTICLES: Set[str] = set()

def is_title(tok: str, titles: Set[str]) -> bool:
    return strip_name_punctuation(tok).lower() in titles

def is_suffix(tok: str, suffixes: Set[str]) -> bool:
    # Сравниваем как есть и без точек (напр. "Ph.D." -> "phd")
    t = strip_name_punctuation(tok).lower()
    t_no_dots = t.replace(".", "")
    return (t in suffixes) or (t_no_dots in suffixes)

def _normalize_for_suffix(s: str) -> str:
    """Нормализация токена/фразы для сравнения с постфиксами: нижний регистр,
    убрать точки и небуквенно-цифровые символы."""
    s = strip_name_punctuation(s).lower()
    s = s.replace(".", "")
    return re.sub(r"[^a-z0-9]+", "", s)

def _strip_trailing_suffixes(parts: list[str], suffixes: Set[str]) -> list[str]:
    """Удаляет с конца списка токены постфиксов, поддерживает комбинации
    из 3/2/1 токенов, нормализуя их к виду без неалфавитных символов.
    Возвращает усечённый список.
    """
    # Подготовим нормализованный сет постфиксов
    suf_norm = {_normalize_for_suffix(x) for x in suffixes}

    i = len(parts)
    while i > 0:
        removed = False
        # Проверяем хвосты 3, 2, 1 токен
        for k in (3, 2, 1):
            if i - k < 0:
                continue
            tail = parts[i - k:i]
            cand = _normalize_for_suffix("".join(tail))
            if cand and cand in suf_norm:
                i -= k
                removed = True
                break
        if not removed:
            break
    return parts[:i]

# ---------- ОСНОВНОЙ АЛГОРИТМ ----------

def extract_name(full_string: str,
                 titles: Set[str],
                 suffixes: Set[str]) -> Tuple[str, str]:
    """
    Возвращает (first_name, last_name).
    Лишние титулы/постфиксы — вырезаются.
    Пробелы — нормализуются и схлопываются.
    Частицы фамилий (von/van/de/...) — сохраняются в фамилии.
    """
    text = normalize_text(full_string)

    # Ранний выход
    if not text:
        return "", ""

    parts = [strip_name_punctuation(p) for p in text.split(" ") if p]

    # Удаляем титулы с начала
    while parts and is_title(parts[0], titles):
        parts.pop(0)

    # Удаляем постфиксы с конца (поддержка много-токенных форм, напр. "Ph D")
    if parts:
        parts = _strip_trailing_suffixes(parts, suffixes)

    if not parts:
        return "", ""

    # Имя — первый токен
    first = parts[0]

    # Фамилия — всё остальное
    # Если второй токен — частица фамилии, оставляем её как часть фамилии
    if len(parts) > 1:
        last_tokens = parts[1:]
        # Нормализуем регистр частиц (часто пишут с маленькой)
        normalized = []
        for i, t in enumerate(last_tokens):
            if t.lower() in LASTNAME_PARTICLES:
                normalized.append(t.lower())
            else:
                normalized.append(t)
        last = " ".join(normalized)
    else:
        last = ""

    return first, last

# ---------- CLI ----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Извлечение чистых имени и фамилии. Словари грузятся из bibl/*.json. По умолчанию читает cleaning/input.txt и пишет cleaning/output.json."
    )
    p.add_argument(
        "--string", type=str, required=False,
        help="Обработать одну строку и вывести результат в консоль."
    )
    p.add_argument(
        "--input", type=Path, default=Path(__file__).parent / "input.txt",
        help="Путь к входному файлу (по умолчанию: cleaning/input.txt)."
    )
    p.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "output.json",
        help="Путь к выходному JSON (по умолчанию: cleaning/output.json)."
    )
    return p

def main():
    args = build_arg_parser().parse_args()

    # Загрузка словарей из bibl/*.json
    titles, suffixes = load_vocab()
    # Загрузка частиц фамилии
    global LASTNAME_PARTICLES
    LASTNAME_PARTICLES = load_lastname_particles()

    # Режим одной строки
    if args.string is not None:
        first, last = extract_name(args.string, titles, suffixes)
        clean_name = (first + (" " + last if last else "")).strip()
        print(f"First Name: {first}")
        print(f"Last  Name: {last}")
        print(f"Clean Name: {clean_name}")
        return

    # Пакетная обработка: построчно из input.txt -> output.json
    input_path: Path = args.input
    output_path: Path = args.output

    items = []
    if input_path.exists():
        try:
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    original = line.rstrip("\n\r")
                    if original.strip() == "":
                        continue
                    first, last = extract_name(original, titles, suffixes)
                    # На всякий случай повторно удалим возможные хвостовые постфиксы из фамилии
                    if last:
                        ltoks = [t for t in last.split() if t]
                        while ltoks and is_suffix(ltoks[-1], suffixes):
                            ltoks.pop()
                        last = " ".join(ltoks)
                    clean_name = (first + (" " + last if last else "")).strip()
                    base_item = {
                        "OriginalName": original,
                        "CleanName": clean_name,
                        "FirstName": first,
                        "LastName": last,
                    }
                    items.append(base_item)
                    # Если фамилия составная (несколько частей), добавляем варианты
                    if last and (" " in last):
                        last_tokens = last.split()
                        last_tail = last_tokens[-1]
                        last_compact = last.replace(" ", "")

                        # Вариант 2: только последняя часть фамилии
                        items.append({
                            "OriginalName": original,
                            "CleanName": (first + (" " + last_tail if last_tail else "")).strip(),
                            "FirstName": first,
                            "LastName": last_tail,
                        })

                        # Вариант 3: вся фамилия без пробелов
                        items.append({
                            "OriginalName": original,
                            "CleanName": (first + (" " + last_compact if last_compact else "")).strip(),
                            "FirstName": first,
                            "LastName": last_compact,
                        })
        except Exception as e:
            print(f"Ошибка чтения входного файла {input_path}: {e}")
    else:
        print(f"Входной файл не найден: {input_path}")
        items = []

    # Запись результатов
    try:
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(items, fout, ensure_ascii=False, indent=2)
        print(f"Записано записей: {len(items)} -> {output_path}")
    except Exception as e:
        print(f"Ошибка записи выходного файла {output_path}: {e}")

if __name__ == "__main__":
    main()