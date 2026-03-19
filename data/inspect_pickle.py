from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import pprint
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config

DEFAULT_PICKLE_PATH = Path(r"D:\Code\data\PPG_FieldStudy\S1\S1.pkl")
FALLBACK_ENCODINGS = ("latin1", "bytes")


def configure_output_streams() -> None:
    """在 Windows 终端中尽量稳定输出 Unicode 文本。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def _resolve_default_pickle_path(config_path: str | Path | None) -> Path:
    cfg = load_config(Path(config_path) if config_path else None)
    configured = str(cfg["path"].get("ppg_fieldstudy_pickle_path", "")).strip()
    return Path(configured).expanduser() if configured else DEFAULT_PICKLE_PATH


def build_parser(default_pickle_path: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 pickle 文件并输出内容")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument(
        "--path",
        type=Path,
        default=default_pickle_path,
        help="待读取的 pickle 文件路径，默认读取 S1.pkl",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="仅输出结构化摘要，避免完整内容过长",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="摘要模式下的递归深度",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=6,
        help="摘要模式下每层最多展示的元素数量",
    )
    parser.add_argument(
        "--max-repr-length",
        type=int,
        default=160,
        help="摘要模式下单个对象预览的最大字符数",
    )
    return parser


def short_repr(value: Any, max_repr_length: int) -> str:
    text = repr(value)
    if len(text) <= max_repr_length:
        return text
    return f"{text[: max_repr_length - 3]}..."


def summarize_value(value: Any, depth: int, max_items: int, max_repr_length: int) -> Any:
    if isinstance(value, (type(None), bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_repr_length else f"{value[: max_repr_length - 3]}..."
    if depth <= 0:
        return {
            "类型": f"{type(value).__module__}.{type(value).__qualname__}",
            "内容预览": short_repr(value, max_repr_length),
        }
    if isinstance(value, dict):
        summary: dict[Any, Any] = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            summary[key] = summarize_value(item, depth - 1, max_items, max_repr_length)
        if len(items) > max_items:
            summary["..."] = f"其余 {len(items) - max_items} 项已省略"
        return summary
    if isinstance(value, (list, tuple)):
        items = [
            summarize_value(item, depth - 1, max_items, max_repr_length)
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"... 其余 {len(value) - max_items} 项已省略")
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, (set, frozenset)):
        items = list(value)
        summary_items = [
            summarize_value(item, depth - 1, max_items, max_repr_length)
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            summary_items.append(f"... 其余 {len(items) - max_items} 项已省略")
        return summary_items
    return {
        "类型": f"{type(value).__module__}.{type(value).__qualname__}",
        "内容预览": short_repr(value, max_repr_length),
    }


def load_pickle_file(path: Path) -> tuple[Any, str]:
    errors: list[str] = []
    try:
        with path.open("rb") as file:
            return pickle.load(file), "default"
    except Exception as exc:
        errors.append(f"default={type(exc).__name__}: {exc}")

    for encoding in FALLBACK_ENCODINGS:
        try:
            with path.open("rb") as file:
                return pickle.load(file, encoding=encoding), encoding
        except Exception as exc:
            errors.append(f"{encoding}={type(exc).__name__}: {exc}")

    joined_errors = "; ".join(errors)
    raise RuntimeError(f"无法读取 pickle 文件 {path}。尝试结果: {joined_errors}")


def main() -> int:
    configure_output_streams()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    default_pickle_path = _resolve_default_pickle_path(pre_args.config)
    args = build_parser(default_pickle_path=default_pickle_path).parse_args()
    path = args.path.expanduser().resolve()

    try:
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        content, encoding_used = load_pickle_file(path)
        print(f"文件路径: {path}")
        print(f"加载方式: {encoding_used}")
        print(f"顶层类型: {type(content).__module__}.{type(content).__qualname__}")
        print("内容:")

        if args.summary:
            output = summarize_value(
                value=content,
                depth=max(args.depth, 0),
                max_items=max(args.max_items, 1),
                max_repr_length=max(args.max_repr_length, 32),
            )
            pprint.pprint(output, sort_dicts=False, width=120)
        else:
            pprint.pprint(content, sort_dicts=False, width=120)
        return 0
    except Exception as exc:
        print(f"读取失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
