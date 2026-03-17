from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np

DEFAULT_PICKLE_PATH = Path(r"D:\Code\data\PPG_FieldStudy\S1\S1.pkl")
ECG_DEFAULT_KEY = ("signal", "chest", "ECG")
PPG_DEFAULT_KEY = ("signal", "wrist", "BVP")
FALLBACK_ENCODINGS = ("latin1", "bytes")

RUN_CONFIG = {
    "path": DEFAULT_PICKLE_PATH,
    "ecg_hz": 700.0,
    "ppg_hz": 64.0,
    "segment_sec": 10.0,
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False,
}


@dataclass(frozen=True)
class SegmentWindow:
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def configure_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Gradio 分页查看 pkl 中的 ECG / PPG 波形")
    parser.add_argument("--path", type=Path, default=RUN_CONFIG["path"], help="pickle 文件路径")
    parser.add_argument("--ecg-hz", type=float, default=RUN_CONFIG["ecg_hz"], help="ECG 采样率")
    parser.add_argument("--ppg-hz", type=float, default=RUN_CONFIG["ppg_hz"], help="PPG 采样率")
    parser.add_argument("--segment-sec", type=float, default=RUN_CONFIG["segment_sec"], help="每页显示的片段时长（秒）")
    parser.add_argument("--server-name", type=str, default=RUN_CONFIG["server_name"], help="Gradio 监听地址")
    parser.add_argument("--server-port", type=int, default=RUN_CONFIG["server_port"], help="Gradio 端口")
    parser.add_argument("--share", action="store_true", default=RUN_CONFIG["share"], help="是否启用 Gradio share")
    return parser


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

    raise RuntimeError(f"无法读取 pickle 文件 {path}。尝试结果: {'; '.join(errors)}")


def get_nested_value(data: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"未找到信号路径: {' -> '.join(keys)}")
        current = current[key]

    array = np.asarray(current)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"信号必须可转换为一维数组，当前 shape={array.shape}")
    return array.astype(np.float32, copy=False)


def get_subject_label(content: dict[str, Any]) -> str:
    subject = content.get("subject")
    if isinstance(subject, (str, int)):
        return str(subject)
    return "未知受试者"


def resolve_selected_path(path_text: str | None, uploaded_file: Any = None) -> Path:
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            candidate = uploaded_file
        elif hasattr(uploaded_file, "name"):
            candidate = str(uploaded_file.name)
        else:
            candidate = str(uploaded_file)
        if candidate:
            return Path(candidate).expanduser().resolve()

    if path_text is None or not str(path_text).strip():
        raise ValueError("请输入 pkl 路径，或先选择一个 pkl 文件。")
    return Path(path_text).expanduser().resolve()


def compute_total_duration_sec(
    ecg_signal: np.ndarray,
    ppg_signal: np.ndarray,
    ecg_hz: float,
    ppg_hz: float,
) -> float:
    if ecg_hz <= 0 or ppg_hz <= 0:
        raise ValueError("采样率必须大于 0")
    return min(float(ecg_signal.shape[0] / ecg_hz), float(ppg_signal.shape[0] / ppg_hz))


def build_segment_windows(total_duration_sec: float, segment_sec: float) -> list[SegmentWindow]:
    if total_duration_sec <= 0:
        raise ValueError("总时长必须大于 0")
    if segment_sec <= 0:
        raise ValueError("segment_sec 必须大于 0")

    segment_count = int(np.ceil(total_duration_sec / segment_sec))
    windows: list[SegmentWindow] = []
    for index in range(segment_count):
        start_sec = index * segment_sec
        end_sec = min(total_duration_sec, start_sec + segment_sec)
        windows.append(SegmentWindow(index=index, start_sec=start_sec, end_sec=end_sec))
    return windows


def slice_signal_for_window(signal: np.ndarray, sample_rate: float, window: SegmentWindow) -> tuple[np.ndarray, np.ndarray]:
    start_index = max(0, int(round(window.start_sec * sample_rate)))
    end_index = min(signal.shape[0], int(round(window.end_sec * sample_rate)))
    sliced = signal[start_index:end_index]
    time_axis = np.arange(sliced.shape[0], dtype=np.float64) / sample_rate + window.start_sec
    return time_axis, sliced


def clamp_page_index(page_index: int, total_pages: int) -> int:
    if total_pages <= 0:
        raise ValueError("total_pages 必须大于 0")
    return max(0, min(page_index, total_pages - 1))


def build_record_bundle(
    path: Path,
    ecg_hz: float,
    ppg_hz: float,
    segment_sec: float,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    content, encoding_used = load_pickle_file(path)
    if not isinstance(content, dict):
        content_type = f"{type(content).__module__}.{type(content).__qualname__}"
        raise TypeError(f"pickle 顶层对象必须是 dict，当前为 {content_type}")

    ecg_signal = get_nested_value(content, ECG_DEFAULT_KEY)
    ppg_signal = get_nested_value(content, PPG_DEFAULT_KEY)
    total_duration_sec = compute_total_duration_sec(ecg_signal, ppg_signal, ecg_hz, ppg_hz)
    windows = build_segment_windows(total_duration_sec=total_duration_sec, segment_sec=segment_sec)
    return {
        "path": str(path),
        "encoding_used": encoding_used,
        "subject_label": get_subject_label(content),
        "ecg_signal": ecg_signal,
        "ppg_signal": ppg_signal,
        "ecg_hz": float(ecg_hz),
        "ppg_hz": float(ppg_hz),
        "segment_sec": float(segment_sec),
        "total_duration_sec": float(total_duration_sec),
        "windows": windows,
    }


def build_summary_text(bundle: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"**受试者**：{bundle['subject_label']}",
            f"**文件路径**：`{bundle['path']}`",
            f"**加载方式**：{bundle['encoding_used']}",
            f"**ECG 采样率**：{bundle['ecg_hz']:.2f} Hz",
            f"**PPG 采样率**：{bundle['ppg_hz']:.2f} Hz",
            f"**对齐后总时长**：{bundle['total_duration_sec'] / 60.0:.2f} 分钟",
            f"**分页长度**：{bundle['segment_sec']:.2f} 秒",
            f"**总页数**：{len(bundle['windows'])}",
        ]
    )


def render_segment_figure(
    ecg_signal: np.ndarray,
    ppg_signal: np.ndarray,
    ecg_hz: float,
    ppg_hz: float,
    windows: list[SegmentWindow],
    page_index: int,
) -> tuple[Figure, str, int]:
    if not windows:
        raise ValueError("windows 不能为空")

    page_index = clamp_page_index(page_index, len(windows))
    window = windows[page_index]

    ecg_time, ecg_slice = slice_signal_for_window(ecg_signal, ecg_hz, window)
    ppg_time, ppg_slice = slice_signal_for_window(ppg_signal, ppg_hz, window)

    fig = Figure(figsize=(12, 6), constrained_layout=True)
    axes = fig.subplots(2, 1, sharex=False)

    axes[0].plot(ecg_time, ecg_slice, color="#b03a2e", linewidth=0.8)
    axes[0].set_title(f"ECG | Segment {page_index + 1}/{len(windows)}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ppg_time, ppg_slice, color="#1f618d", linewidth=0.8)
    axes[1].set_title("PPG / BVP")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.2)

    info = (
        f"第 **{page_index + 1} / {len(windows)}** 段 | "
        f"时间窗 **{window.start_sec:.2f}s - {window.end_sec:.2f}s** | "
        f"时长 **{window.duration_sec:.2f}s** | "
        f"ECG 点数 **{ecg_slice.shape[0]}** | "
        f"PPG 点数 **{ppg_slice.shape[0]}**"
    )
    return fig, info, page_index


def jump_to_page(
    requested_page: float,
    bundle: dict[str, Any],
) -> tuple[Figure, str, int, int]:
    requested_index = int(round(requested_page)) - 1
    fig, info, page_index = render_segment_figure(
        ecg_signal=bundle["ecg_signal"],
        ppg_signal=bundle["ppg_signal"],
        ecg_hz=bundle["ecg_hz"],
        ppg_hz=bundle["ppg_hz"],
        windows=bundle["windows"],
        page_index=requested_index,
    )
    return fig, info, page_index + 1, page_index


def step_page(
    current_index: int,
    delta: int,
    bundle: dict[str, Any],
) -> tuple[Figure, str, int, int]:
    return jump_to_page(requested_page=current_index + delta + 1, bundle=bundle)


def load_record_for_display(
    path_text: str | None,
    uploaded_file: Any,
    ecg_hz: float,
    ppg_hz: float,
    segment_sec: float,
) -> tuple[Figure, str, int, int, dict[str, Any], str, str, str]:
    path = resolve_selected_path(path_text, uploaded_file)
    bundle = build_record_bundle(path=path, ecg_hz=ecg_hz, ppg_hz=ppg_hz, segment_sec=segment_sec)
    fig, info, page_number, page_index = jump_to_page(requested_page=1, bundle=bundle)
    summary_text = build_summary_text(bundle)
    status_text = f"已载入：`{path}`"
    return fig, info, page_number, page_index, bundle, summary_text, status_text, str(path)


def create_gradio_app(initial_bundle: dict[str, Any]):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("未安装 gradio，请先安装 gradio 后再运行该脚本。") from exc

    initial_fig, initial_info, initial_page_number, initial_index = jump_to_page(
        requested_page=1,
        bundle=initial_bundle,
    )
    summary_text = build_summary_text(initial_bundle)
    initial_path = str(initial_bundle["path"])
    initial_ecg_hz = float(initial_bundle["ecg_hz"])
    initial_ppg_hz = float(initial_bundle["ppg_hz"])
    initial_segment_sec = float(initial_bundle["segment_sec"])

    with gr.Blocks(title="ECG / PPG 分页波形查看器") as demo:
        gr.Markdown("# ECG / PPG 分页波形查看器")
        gr.Markdown(
            "按固定时间窗浏览整段 ECG / PPG 波形，可通过页码、上一页和下一页快速检查长时记录。"
        )
        summary_markdown = gr.Markdown(summary_text)

        with gr.Row():
            path_input = gr.Textbox(label="PKL 路径", value=initial_path, scale=4)
            file_input = gr.File(label="选择 PKL 文件", file_types=[".pkl"], file_count="single")
            load_button = gr.Button("载入波形", variant="primary")

        status_markdown = gr.Markdown("已载入默认记录。")
        data_state = gr.State(value=initial_bundle)
        page_state = gr.State(value=initial_index)

        with gr.Row():
            prev_button = gr.Button("上一页")
            next_button = gr.Button("下一页")
            page_input = gr.Number(
                label="页码（从 1 开始）",
                value=initial_page_number,
                minimum=1,
                maximum=len(initial_bundle["windows"]),
                precision=0,
            )
            jump_button = gr.Button("跳转")

        plot = gr.Plot(value=initial_fig, label="当前 10 秒片段")
        info_markdown = gr.Markdown(initial_info)

        file_input.upload(
            fn=lambda file_obj: str(resolve_selected_path(None, file_obj)) if file_obj is not None else "",
            inputs=file_input,
            outputs=path_input,
        )

        load_button.click(
            fn=lambda path_text, uploaded_file: load_record_for_display(
                path_text=path_text,
                uploaded_file=uploaded_file,
                ecg_hz=initial_ecg_hz,
                ppg_hz=initial_ppg_hz,
                segment_sec=initial_segment_sec,
            ),
            inputs=[path_input, file_input],
            outputs=[
                plot,
                info_markdown,
                page_input,
                page_state,
                data_state,
                summary_markdown,
                status_markdown,
                path_input,
            ],
        )

        prev_button.click(
            fn=lambda idx, bundle: step_page(idx, -1, bundle),
            inputs=[page_state, data_state],
            outputs=[plot, info_markdown, page_input, page_state],
        )
        next_button.click(
            fn=lambda idx, bundle: step_page(idx, 1, bundle),
            inputs=[page_state, data_state],
            outputs=[plot, info_markdown, page_input, page_state],
        )
        jump_button.click(
            fn=lambda page, bundle: jump_to_page(page, bundle),
            inputs=[page_input, data_state],
            outputs=[plot, info_markdown, page_input, page_state],
        )

    return demo


def main() -> int:
    configure_output_streams()
    args = build_parser().parse_args()
    path = args.path.expanduser().resolve()

    try:
        initial_bundle = build_record_bundle(
            path=path,
            ecg_hz=args.ecg_hz,
            ppg_hz=args.ppg_hz,
            segment_sec=args.segment_sec,
        )
        demo = create_gradio_app(initial_bundle=initial_bundle)

        print(f"文件路径: {path}")
        print(f"加载方式: {initial_bundle['encoding_used']}")
        print(f"ECG 长度: {initial_bundle['ecg_signal'].shape[0]} | 采样率: {args.ecg_hz} Hz")
        print(f"PPG 长度: {initial_bundle['ppg_signal'].shape[0]} | 采样率: {args.ppg_hz} Hz")
        print(f"分页时长: {args.segment_sec} 秒")
        print(f"监听地址: http://{args.server_name}:{args.server_port}")

        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            inbrowser=False,
        )
        return 0
    except Exception as exc:
        print(f"启动 Gradio 波形查看器失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
