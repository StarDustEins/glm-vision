from __future__ import annotations

import os
from types import SimpleNamespace
import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw

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


def _extract_structured_answer(
    answer_text: str,
) -> Tuple[str, List[Dict[str, float]]]:
    """将模型返回的 JSON 答案转换为 Markdown，同时提取标注框。"""
    cleaned = (answer_text or "").strip()
    if not cleaned:
        return "", []

    json_payload: Optional[Dict[str, Any]] = None
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            json_payload = json.loads(cleaned)
        except json.JSONDecodeError:
            json_payload = None
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                json_payload = json.loads(candidate)
            except json.JSONDecodeError:
                json_payload = None

    if not isinstance(json_payload, dict):
        return cleaned, []

    markers_field = json_payload.get("makers")
    if markers_field is None:
        markers_field = json_payload.get("markers")
    if markers_field is None:
        markers_field = []
    markers: List[Dict[str, float]] = []
    if isinstance(markers_field, list):
        for item in markers_field:
            if not isinstance(item, dict):
                continue
            try:
                height_value = item.get("height", item.get("hight"))
                marker = {
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "width": float(item["width"]),
                    "height": float(height_value),
                }
            except (KeyError, TypeError, ValueError):
                continue
            markers.append(marker)

    section_map = {
        "部位信息": json_payload.get("部位信息", ""),
        "检查所见": json_payload.get("检查所见", ""),
        "诊断意见": json_payload.get("诊断意见", ""),
        "置信度": json_payload.get("置信度", ""),
    }

    lines = [
        f"1. **部位信息**：{section_map['部位信息']}",
        f"2. **检查所见**：{section_map['检查所见']}",
        f"3. **诊断意见**：{section_map['诊断意见']}",
        f"4. **置信度**：{section_map['置信度']}",
    ]

    if markers:
        lines.append(
            "5. **makers**："
            + ", ".join(
                f"(x={m['x']:.3f}, y={m['y']:.3f}, w={m['width']:.3f}, h={m['height']:.3f})"
                for m in markers
            )
        )
    else:
        lines.append("5. **makers**：[]")

    return "\n".join(lines), markers


def _build_slider_images(
    base_image: Optional[Image.Image],
    markers: List[Dict[str, float]],
) -> Optional[Tuple[Image.Image, Image.Image]]:
    """构建原图与标注图对，供 ImageSlider 使用。"""
    if base_image is None:
        return None

    original = base_image.copy()
    annotated = base_image.copy()

    if not markers:
        return (original, annotated)

    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    has_box = False

    for marker in markers:
        try:
            x = float(marker["x"])
            y = float(marker["y"])
            w = float(marker["width"])
            h = float(marker["height"])
        except (KeyError, TypeError, ValueError):
            continue

        x0 = max(0.0, min(1.0, x)) * width
        y0 = max(0.0, min(1.0, y)) * height
        x1 = max(0.0, min(1.0, x + w)) * width
        y1 = max(0.0, min(1.0, y + h)) * height

        if x1 <= x0 or y1 <= y0:
            continue

        draw.rectangle([x0, y0, x1, y1], outline="#ff4d4f", width=4)
        has_box = True

    if not has_box:
        return (original, original.copy())

    return (original, annotated)


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
) -> Tuple[
    Dict[str, Any],
    str,
    str,
    str,
    pd.DataFrame,
    str,
    Optional[Tuple[Image.Image, Image.Image]],
]:
    state = dict(state or {})

    def _final(
        status_html: str,
        answer: str = "",
        think: str = "",
        slider_images: Optional[Tuple[Image.Image, Image.Image]] = None,
    ) -> Tuple[
        Dict[str, Any],
        str,
        str,
        str,
        pd.DataFrame,
        str,
        Optional[Tuple[Image.Image, Image.Image]],
    ]:
        return (
            state,
            status_html,
            answer,
            think,
            _build_history_dataframe(),
            _format_cache_metrics(),
            slider_images,
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

    display_image: Optional[Image.Image] = None
    processing_image: Optional[Image.Image] = None

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
            display_image = img.convert("RGB")
        processing_image = display_image.copy()
    except Exception as exc:  # noqa: BLE001
        return _final(
            _compose_status(f"❌ 图像读取失败：{exc}", level="error"),
        )

    try:
        result = process_image_analysis(
            image=processing_image,
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
    raw_answer = result.answer_content or result.raw_result
    formatted_answer, markers = _extract_structured_answer(raw_answer)
    slider_images = _build_slider_images(display_image, markers)
    think = result.think_content or "（模型未返回显性推理过程）"

    history_df = _build_history_dataframe()
    cache_stats = _format_cache_metrics()

    return state, status, formatted_answer, think, history_df, cache_stats, slider_images


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
    ) -> Tuple[
        Dict[str, Any],
        str,
        str,
        str,
        pd.DataFrame,
        str,
        Optional[Tuple[Image.Image, Image.Image]],
    ]:
        (
            next_state,
            status_html,
            answer_md,
            think_md,
            history_df,
            cache_stats,
            slider_images,
        ) = _run_analysis(
            selected_model,
            image_path,
            use_cache,
            state,
            available_models,
        )
        slider_update = (
            gr.update(value=slider_images) if slider_images is not None else gr.update(value=None)
        )
        return (
            next_state,
            status_html,
            answer_md,
            think_md,
            history_df,
            cache_stats,
            slider_update,
        )

    def _on_analysis_start() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """推理开始时禁用按钮并提示加载状态。"""
        return (
            gr.update(interactive=False, value="分析中..."),
            gr.update(
                value=_compose_status("⏳ 正在分析图像，请稍候...", level="info")
            ),
            gr.update(value=None),
        )

    def _on_analysis_end() -> Dict[str, Any]:
        """推理结束后恢复按钮可用状态。"""
        return gr.update(interactive=True, value="开始分析")

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
                        image_slider = gr.ImageSlider(
                            label="原始图像 / 标注对比",
                            value=None,
                            interactive=False,
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

                        image_input = gr.File(
                            label="上传医学影像文件",
                            file_types=["image"],
                            file_count="single",
                            type="filepath",
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
            outputs=[analyze_button, status_display, image_slider],
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
                image_slider,
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
