from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import gradio as gr
import pandas as pd
from PIL import Image

from app_logic import (
    DEFAULT_MEDICAL_PROMPT,
    AnalysisResult,
    build_available_models,
    fetch_cache_metrics,
    get_analysis_history,
    get_device_info,
    init_database,
    load_model,
    process_image_analysis,
    validate_image,
)


def _init_state() -> Dict[str, Any]:
    return {
        "model": None,
        "processor": None,
        "device": None,
        "model_type": None,
        "name": None,
    }


def _format_device_info() -> str:
    info = get_device_info()
    if not info:
        return "### 硬件信息\n- 未检测到设备信息"

    lines = ["### 硬件信息"]
    for item in info:
        label = item.get("label", "设备")
        value = item.get("value", "未知")
        help_text = item.get("help")
        if help_text:
            lines.append(f"- **{label}**：{value}\\n  - {help_text}")
        else:
            lines.append(f"- **{label}**：{value}")
    return "\n".join(lines)


def _format_cache_metrics() -> str:
    metrics = fetch_cache_metrics()
    total = metrics.get("total_records", 0)
    recent = metrics.get("recent_records", 0)
    db_size = metrics.get("db_size_mb", 0.0)
    lines = ["### 缓存概览"]
    lines.append(f"- 总缓存条目：{total}")
    lines.append(f"- 最近7天分析：{recent}")
    lines.append(f"- 数据库大小：{db_size:.2f} MB")
    return "\n".join(lines)


def _build_history_dataframe(limit: int = 20) -> pd.DataFrame:
    records = get_analysis_history(limit=limit)
    if not records:
        return pd.DataFrame(
            columns=["生成时间", "模型名称", "耗时(s)", "图像尺寸", "诊断摘要"]
        )

    rows = []
    for record in records:
        summary = record.get("answer_content") or record.get("think_content") or ""
        summary = summary.replace("\n", " ")
        if len(summary) > 120:
            summary = summary[:117] + "..."
        rows.append(
            {
                "生成时间": record.get("created_at"),
                "模型名称": record.get("model_name"),
                "耗时(s)": round(float(record.get("processing_time", 0.0)), 2),
                "图像尺寸": record.get("image_size"),
                "诊断摘要": summary,
            }
        )

    return pd.DataFrame(rows)


def _compose_status(message: str, level: str = "info") -> str:
    level_class = {
        "info": "status-info",
        "warning": "status-warning",
        "error": "status-error",
        "success": "status-success",
    }.get(level, "status-info")
    return f"<div class='status-line {level_class}'>{message}</div>"


def _load_selected_model(
    model_name: str,
    state: Dict[str, Any],
    available_models: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Any], str]:
    state = dict(state or {})
    if not model_name:
        return state, "⚠️ 请先选择模型再加载。"

    config = available_models.get(model_name)
    if config is None:
        return state, f"⚠️ 未找到模型：{model_name}"

    try:
        model_bundle, processor, device, model_type = load_model(model_name, config)
    except Exception as exc:  # noqa: BLE001
        return state, f"❌ 模型加载失败：{exc}"

    new_state = {
        "model": model_bundle,
        "processor": processor,
        "device": device,
        "model_type": model_type,
        "name": model_name,
    }

    device_repr = str(device)
    status = f"✅ 模型已加载：{model_name} （类型：{model_type}，设备：{device_repr}）"
    return new_state, status


def _run_analysis(
    model_name: str,
    image_path: str,
    enable_cache: bool,
    state: Dict[str, Any],
    available_models: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Any], str, str, str, pd.DataFrame, str]:
    state = dict(state or {})

    def _final(
        status_html: str, answer: str = "", think: str = ""
    ) -> Tuple[
        Dict[str, Any],
        str,
        str,
        str,
        pd.DataFrame,
        str,
    ]:
        return (
            state,
            status_html,
            answer,
            think,
            _build_history_dataframe(),
            _format_cache_metrics(),
        )

    if not available_models:
        return _final(_compose_status("❌ 未发现可用模型，请检查配置。", level="error"))

    if not model_name:
        return _final(
            _compose_status("⚠️ 请先选择模型。", level="warning"),
        )

    if model_name not in available_models:
        return _final(
            _compose_status(f"❌ 未找到模型：{model_name}", level="error"),
        )

    if not image_path:
        return _final(
            _compose_status("⚠️ 请上传医学影像文件。", level="warning"),
        )

    if state.get("name") != model_name or not state.get("model"):
        state, load_message = _load_selected_model(
            model_name,
            state,
            available_models,
        )
        state = dict(state or {})
        if not state.get("model"):
            return _final(
                _compose_status(str(load_message or "模型加载失败。"), level="error")
            )

    prompt_text = DEFAULT_MEDICAL_PROMPT.strip()

    try:
        file_size = os.path.getsize(image_path)
        filename = os.path.basename(image_path)
        file_stub = SimpleNamespace(name=filename, size=file_size)
        is_valid, message = validate_image(file_stub)
        if not is_valid:
            return _final(
                _compose_status(f"⚠️ {message}", level="warning"),
            )
        with Image.open(image_path) as img:
            image = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return _final(
            _compose_status(f"❌ 图像读取失败：{exc}", level="error"),
        )

    try:
        result = process_image_analysis(
            image=image,
            model=state["model"],
            processor=state.get("processor"),
            device=state.get("device"),
            model_type=state.get("model_type", ""),
            model_name=state.get("name", ""),
            image_name=filename,
            prompt=prompt_text,
            enable_cache=enable_cache,
        )
    except Exception as exc:  # noqa: BLE001
        return _final(
            _compose_status(f"❌ 推理过程出现错误：{exc}", level="error"),
        )

    status = _render_status(result)
    answer = result.answer_content or result.raw_result
    think = result.think_content or "（模型未返回显性推理过程）"

    history_df = _build_history_dataframe()
    cache_stats = _format_cache_metrics()

    return state, status, answer, think, history_df, cache_stats


def _render_status(result: AnalysisResult) -> str:
    flag = "缓存" if result.from_cache else "实时推理"
    created = result.created_at or "未知时间"
    flag_class = "status-cache" if result.from_cache else "status-live"
    message = (
        "✅ <span class='status-label'>完成</span> "
        f"<span class='{flag_class}'>{flag}</span>"
        f"<span class='status-body'>：耗时 {result.processing_time:.2f}s，生成时间 {created}</span>"
    )
    return _compose_status(message, level="success")


def _refresh_history_table() -> pd.DataFrame:
    return _build_history_dataframe()


def _refresh_cache_panel() -> str:
    return _format_cache_metrics()


def main() -> gr.Blocks:
    init_database()
    available_models = build_available_models()

    default_model = next(iter(available_models.keys()), None)
    history_df = _build_history_dataframe()
    cache_stats = _format_cache_metrics()

    theme = gr.themes.Soft()

    def _handle_analysis(
        selected_model: str,
        image_path: str,
        use_cache: bool,
        state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, str, str, pd.DataFrame, str]:
        return _run_analysis(
            selected_model,
            image_path,
            use_cache,
            state,
            available_models,
        )

    def _on_analysis_start() -> Tuple[gr.Button, gr.HTML]:
        """推理开始时禁用按钮并提示加载状态。"""
        return (
            gr.Button.update(interactive=False),
            gr.update(
                value=_compose_status("⏳ 正在分析图像，请稍候...", level="info")
            ),
        )

    def _on_analysis_end() -> gr.Button:
        """推理结束后恢复按钮可用状态。"""
        return gr.Button.update(interactive=True)

    with gr.Blocks(
        title="医学图像AI诊断系统",
        theme=theme,
    ) as demo:
        gr.Markdown("# 医学图像AI诊断系统")
        gr.Markdown("结合本地或云端模型，快速完成医学影像的自动诊断分析。")

        model_state = gr.State(_init_state())

        with gr.Tabs(elem_id="page-tabs"):
            with gr.Tab("业务"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, elem_id="inference-panel"):
                        # gr.Markdown("### 上传与结果")
                        image_input = gr.Image(
                            label="医学影像",
                            type="filepath",
                            sources=["upload"],
                            image_mode="RGB",
                            height=420,
                            elem_id="preview-image",
                        )

                        status_display = gr.HTML(
                            _compose_status(
                                "📌 等待分析，请先上传图像。", level="info"
                            ),
                            elem_id="status-box",
                        )

                    with gr.Column(scale=1, elem_id="control-panel"):
                        # gr.Markdown("### 模型与参数")
                        model_dropdown = gr.Dropdown(
                            choices=list(available_models.keys()),
                            value=default_model,
                            label="选择模型",
                            interactive=bool(available_models),
                        )


                        cache_checkbox = gr.Checkbox(label="启用缓存", value=True)
                        analyze_button = gr.Button("开始分析", variant="primary")

                        if analyze_button is not None:
                            with gr.Tabs(elem_id="result-tabs"):
                                with gr.Tab("诊断结果"):
                                    answer_md = gr.Markdown("尚未生成诊断内容。")
                                with gr.Tab("推理过程"):
                                    think_md = gr.Markdown("暂无推理内容。")

            with gr.Tab("历史"):
                history_refresh = gr.Button("刷新历史", variant="secondary")
                history_table = gr.Dataframe(
                    value=history_df,
                    label="最近分析记录",
                    interactive=False,
                )

            with gr.Tab("缓存"):
                cache_refresh = gr.Button("刷新缓存", variant="secondary")
                cache_md = gr.Markdown(cache_stats)

        analyze_button.click(
            fn=_on_analysis_start,
            inputs=None,
            outputs=[analyze_button, status_display],
            queue=False,
        ).then(
            fn=_handle_analysis,
            inputs=[model_dropdown, image_input, cache_checkbox, model_state],
            outputs=[
                model_state,
                status_display,
                answer_md,
                think_md,
                history_table,
                cache_md,
            ],
        ).then(
            fn=_on_analysis_end,
            inputs=None,
            outputs=analyze_button,
            queue=False,
        )

        history_refresh.click(
            fn=_refresh_history_table,
            inputs=None,
            outputs=history_table,
        )

        cache_refresh.click(
            fn=_refresh_cache_panel,
            inputs=None,
            outputs=cache_md,
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(
        pwa=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
