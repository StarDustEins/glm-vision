from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List

import streamlit as st
from PIL import Image

from app_logic import (
    AnalysisResult,
    DEFAULT_MEDICAL_PROMPT,
    SUPPORTED_FORMATS,
    build_available_models,
    clear_analysis_cache,
    fetch_all_cache_records,
    fetch_cache_metrics,
    get_analysis_history,
    get_device_info,
    init_database,
    load_model,
    process_image_analysis,
    validate_image,
)


st.set_page_config(
    page_title="医学图像AI诊断系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _ensure_session_state(available_models: Dict[str, Dict[str, str]]) -> None:
    """初始化 Streamlit session 状态。"""
    defaults = {
        "model_loaded": False,
        "model": None,
        "processor": None,
        "device": None,
        "model_type": "glm4v",
        "analysis_results": [],
        "loading_model": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.session_state.available_models = available_models

    if not available_models:
        st.session_state.current_model = None
        st.session_state.model_loaded = False
        return

    if (
        "current_model" not in st.session_state
        or st.session_state.current_model not in available_models
    ):
        st.session_state.current_model = next(iter(available_models.keys()))
        st.session_state.model_loaded = False


def _register_analysis_result(
    filename: str,
    image: Image.Image,
    result: AnalysisResult,
) -> Dict[str, str]:
    """将分析结果存入 session_state，便于历史记录展示。"""
    record = {
        "filename": filename,
        "timestamp": result.created_at,
        "image_size": f"{image.size[0]}x{image.size[1]}",
        "processing_time": f"{result.processing_time:.2f}s",
        "raw_result": result.raw_result,
        "think_content": result.think_content,
        "answer_content": result.answer_content,
        "status": "success",
        "from_cache": result.from_cache,
        "model_name": st.session_state.current_model,
    }
    st.session_state.analysis_results.append(record)
    return record


def _display_analysis_panels(result: AnalysisResult) -> None:
    """在 UI 中展示分析结果。"""
    st.markdown(f"**⏱️ 处理时间：** {result.processing_time:.2f} 秒")

    if result.answer_content:
        with st.expander("📋 诊断结果", expanded=True):
            st.write(result.answer_content)

    if result.think_content:
        with st.expander("🤔 分析过程", expanded=False):
            st.write(result.think_content)

    if not result.answer_content and not result.think_content:
        st.write("**诊断结果：**")
        st.write(result.raw_result)


def _build_download_report(
    filename: str, image: Image.Image, result: AnalysisResult
) -> str:
    """构建下载报告文本。"""
    report = ["医学图像诊断报告", "=" * 50]
    report.append(f"文件名：{filename}")
    report.append(f"生成时间：{result.created_at}")
    if st.session_state.current_model:
        report.append(f"使用模型：{st.session_state.current_model}")
    report.append(f"处理时间：{result.processing_time:.2f} 秒")
    report.append(f"图像尺寸：{image.size[0]}x{image.size[1]} 像素\n")

    if result.answer_content:
        report.append("诊断结果：")
        report.append("-" * 30)
        report.append(result.answer_content)
        report.append("")

    if result.think_content:
        report.append("分析过程：")
        report.append("-" * 30)
        report.append(result.think_content)
        report.append("")

    if not result.answer_content and not result.think_content:
        report.append("诊断结果：")
        report.append(result.raw_result)

    return "\n".join(report)


def _create_stream_callback(placeholder):
    """构建用于流式展示的回调。"""
    buffer: List[str] = []

    def _callback(chunk: str) -> None:
        buffer.append(chunk)
        placeholder.write("".join(buffer))

    return _callback, buffer


def _ensure_card_style() -> None:
    st.markdown(
        """
        <style>
        .info-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(140, 140, 140, 0.18);
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.6rem;
        }
        .info-card .info-label {
            color: var(--secondary-text-color, #5e6a7d);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.1rem;
        }
        .info-card .info-value {
            font-size: 1.08rem;
            font-weight: 600;
            color: var(--text-color, inherit);
            word-break: break-word;
        }
        .info-card .info-help {
            color: var(--secondary-text-color, #7a8796);
            font-size: 0.72rem;
            margin-top: 0.25rem;
        }
        .info-card code {
            background: rgba(255,255,255,0.12);
            padding: 0.05rem 0.35rem;
            border-radius: 4px;
            font-size: 0.82rem;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_info_cards(items: List[Dict[str, str]], columns: int = 1) -> None:
    _ensure_card_style()

    if not items:
        st.info("暂无信息")
        return

    for start in range(0, len(items), columns):
        current = items[start : start + columns]
        cols = st.columns(len(current))
        for col, item in zip(cols, current):
            value = str(item.get("value", "-"))
            help_text = item.get("help", "")
            col.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">{item.get("label", "")}</div>
                    <div class="info-value">{value}</div>
                    {f'<div class="info-help">{help_text}</div>' if help_text else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _load_selected_model(selected_model: str, model_config: Dict[str, str]) -> None:
    """加载选定的模型并更新 session 状态。"""
    with st.spinner(f"正在加载 {selected_model} 模型，请稍候..."):
        model, processor, device, model_type = load_model(selected_model, model_config)

    st.session_state.model = model
    st.session_state.processor = processor
    st.session_state.device = device
    st.session_state.current_model = selected_model
    st.session_state.model_type = model_type
    st.session_state.model_loaded = True
    st.session_state.loading_model = False
    st.session_state.pop("loading_target", None)
    st.success(f"✅ {selected_model} 模型加载成功！")
    st.rerun()


def _render_device_info_section() -> None:
    device_info = get_device_info()
    _render_info_cards(device_info, columns=1)


def _render_sidebar() -> None:
    """渲染侧边栏配置。"""
    available_models = st.session_state.available_models

    def _format_model_option(name: str) -> str:
        info = available_models[name]
        if info.get("source") == "local":
            identifier = info.get("identifier", name)
            if "/" in identifier:
                owner, model = identifier.split("/", 1)
                return f"{owner} / {model}"
            return identifier
        return info.get("description", name)

    with st.sidebar:
        st.header("🔧 系统配置")

        st.subheader("💻 设备信息")
        _render_device_info_section()

        st.divider()

        st.subheader("🤖 模型")

        if not available_models:
            st.error("⚠️ 未发现可用模型，请检查默认目录或网络配置")
            return

        model_names = list(available_models.keys())
        default_index = (
            model_names.index(st.session_state.current_model)
            if st.session_state.current_model in model_names
            else 0
        )

        selected_model = st.selectbox(
            "选择模型",
            options=model_names,
            index=default_index,
            format_func=_format_model_option,
        )

        model_info = available_models[selected_model]
        st.caption(f"模型位置：{model_info.get('path')}")

        model_needs_reload = (
            not st.session_state.model_loaded
            or st.session_state.current_model != selected_model
        )

        is_loading = st.session_state.get("loading_model", False)
        loading_target = st.session_state.get("loading_target")

        if is_loading and loading_target and loading_target != selected_model:
            st.session_state.loading_model = False
            st.session_state.pop("loading_target", None)
            st.info("已切换模型，取消先前的加载任务")
            is_loading = False

        button_label = "⏳ 正在加载...点击取消" if is_loading else (
            "🚀 加载模型" if model_needs_reload else "🔄 重新加载模型"
        )

        button_clicked = st.button(
            button_label,
            key="load_model_button",
            width="stretch",
            type="primary",
        )

        if button_clicked:
            if is_loading:
                st.session_state.loading_model = False
                st.session_state.pop("loading_target", None)
                st.info("已取消模型加载")
            else:
                st.session_state.loading_model = True
                st.session_state.loading_target = selected_model
                try:
                    _load_selected_model(selected_model, model_info)
                except Exception as exc:
                    st.session_state.loading_model = False
                    st.session_state.pop("loading_target", None)
                    st.error(f"❌ 模型加载失败：{exc}")

        if st.session_state.model_loaded and (
            st.session_state.current_model == selected_model
        ):
            st.success(f"{selected_model} 已加载")
        else:
            st.warning("模型尚未加载")

        st.divider()


def _render_single_analysis_tab() -> None:
    st.header("单张图像分析")

    uploaded_file = st.file_uploader(
        "选择医学图像文件",
        type=[ext.strip(".") for ext in SUPPORTED_FORMATS],
        help="支持 JPG/JPEG/PNG/BMP/TIFF/WebP，大小限制 10MB",
    )

    if uploaded_file is None:
        return

    is_valid, message = validate_image(uploaded_file)
    if not is_valid:
        st.error(f"❌ {message}")
        return

    st.success(f"✅ {message}")

    col_image, col_action = st.columns([1, 1])
    with col_image:
        st.subheader("📷 原始图像")
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            image = img.convert("RGB")
        st.image(
            image,
            caption=f"文件名: {uploaded_file.name}",
            width="stretch",
        )

        st.markdown(
            f"""
            <div class="image-info">
            📏 <strong>尺寸：</strong> {image.size[0]} × {image.size[1]} 像素<br>
            🎨 <strong>模式：</strong> {image.mode}<br>
            📝 <strong>格式：</strong> {uploaded_file.type or "N/A"}<br>
            📁 <strong>大小：</strong> {uploaded_file.size / 1024:.1f} KB
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_action:
        st.subheader("🤖 AI诊断分析")
        start_analysis = st.button("🔍 开始分析", type="primary")
        streaming_placeholder = st.empty()

        if start_analysis:
            if not st.session_state.model_loaded:
                st.error("⚠️ 请先在侧边栏加载模型")
                return

            stream_callback, stream_buffer = _create_stream_callback(
                streaming_placeholder
            )

            try:
                analysis_result = process_image_analysis(
                    image=image,
                    model=st.session_state.model,
                    processor=st.session_state.processor,
                    device=st.session_state.device,
                    model_type=st.session_state.model_type,
                    model_name=st.session_state.current_model,
                    image_name=uploaded_file.name,
                    prompt=DEFAULT_MEDICAL_PROMPT,
                    enable_cache=False,
                    stream_callback=stream_callback,
                )
            except Exception as exc:
                streaming_placeholder.empty()
                st.error(f"❌ 分析失败：{exc}")
                return

            streaming_placeholder.empty()

            if analysis_result.from_cache:
                st.info("🎯 命中缓存：已加载历史分析结果")

            _display_analysis_panels(analysis_result)
            _register_analysis_result(uploaded_file.name, image, analysis_result)

            download_data = _build_download_report(
                uploaded_file.name, image, analysis_result
            )
            st.download_button(
                label="📥 下载诊断报告",
                data=download_data,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_诊断报告.txt",
                mime="text/plain",
            )


def _render_batch_analysis_tab() -> None:
    st.header("批量图像处理")
    uploaded_files = st.file_uploader(
        "选择多张医学图像文件",
        type=[ext.strip(".") for ext in SUPPORTED_FORMATS],
        accept_multiple_files=True,
        help="可以同时上传多张图像进行批量分析",
    )

    if not uploaded_files:
        return

    st.success(f"✅ 已选择 {len(uploaded_files)} 个文件")
    with st.expander("📁 文件列表", expanded=True):
        for index, file in enumerate(uploaded_files, start=1):
            st.write(f"{index}. {file.name} ({file.size / 1024:.1f} KB)")

    if not st.button("🚀 开始批量分析", type="primary"):
        return

    if not st.session_state.model_loaded:
        st.error("⚠️ 请先在侧边栏加载模型")
        return

    overall_progress = st.progress(0.0)
    status_placeholder = st.empty()
    results_container = st.container()

    batch_results: List[Dict[str, str]] = []
    total_start_time = time.time()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        progress = (index - 1) / len(uploaded_files)
        overall_progress.progress(progress)
        status_placeholder.write(
            f"🔄 正在处理第 {index}/{len(uploaded_files)} 个文件：{uploaded_file.name}"
        )

        is_valid, message = validate_image(uploaded_file)
        if not is_valid:
            warning_message = f"⚠️ {uploaded_file.name}: {message}"
            batch_results.append(
                {
                    "filename": uploaded_file.name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "invalid",
                    "error": message,
                    "model_name": st.session_state.current_model,
                }
            )
            with results_container:
                st.warning(warning_message)
            continue

        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            image = img.convert("RGB")

        try:
            analysis_result = process_image_analysis(
                image=image,
                model=st.session_state.model,
                processor=st.session_state.processor,
                device=st.session_state.device,
                model_type=st.session_state.model_type,
                model_name=st.session_state.current_model,
                image_name=uploaded_file.name,
                prompt=DEFAULT_MEDICAL_PROMPT,
                enable_cache=True,
                stream_callback=None,
            )
        except Exception as exc:
            error_message = f"❌ {uploaded_file.name} 处理失败：{exc}"
            batch_results.append(
                {
                    "filename": uploaded_file.name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "error",
                    "error": str(exc),
                    "model_name": st.session_state.current_model,
                }
            )
            with results_container:
                st.error(error_message)
            continue

        record = _register_analysis_result(uploaded_file.name, image, analysis_result)
        batch_results.append(record)

        with results_container:
            header = f"✅ {uploaded_file.name} - 分析完成 ({analysis_result.processing_time:.2f}s)"
            if analysis_result.from_cache:
                header += " [缓存]"
            with st.expander(header, expanded=False):
                col_img, col_result = st.columns([1, 2])
                with col_img:
                    st.image(
                        image,
                        caption=uploaded_file.name,
                        width="stretch",
                    )
                with col_result:
                    _display_analysis_panels(analysis_result)

    overall_progress.progress(1.0)
    total_time = time.time() - total_start_time
    status_placeholder.success(
        f"🎉 批量处理完成！总耗时：{total_time:.2f} 秒（平均每张 {total_time / max(len(uploaded_files), 1):.2f} 秒）"
    )

    if batch_results:
        report_lines = [
            "批量医学图像分析报告",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"总文件数：{len(uploaded_files)}",
            f"总处理时间：{total_time:.2f} 秒",
            f"平均处理时间：{total_time / max(len(uploaded_files), 1):.2f} 秒",
            "=" * 80,
            "",
        ]

        for item in batch_results:
            report_lines.append(f"文件名：{item['filename']}")
            report_lines.append(f"时间戳：{item['timestamp']}")
            if item.get("model_name"):
                report_lines.append(f"模型：{item['model_name']}")
            if item.get("status") == "success":
                report_lines.append(f"图像尺寸：{item.get('image_size', '未知')}")
                report_lines.append(f"处理时间：{item.get('processing_time', 'N/A')}")
                report_lines.append("诊断结果：")
                report_lines.append(item.get("raw_result", ""))
            else:
                report_lines.append(f"状态：{item.get('status')}")
                report_lines.append(f"错误：{item.get('error', '未知错误')}")
            report_lines.append("\n" + "-" * 80 + "\n")

        st.download_button(
            label="📦 下载批量分析报告",
            data="\n".join(report_lines),
            file_name=f"批量分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


def _render_history_tab() -> None:
    st.header("历史分析结果")
    results = st.session_state.analysis_results

    if not results:
        st.info("📭 暂无历史分析结果")
        return

    total_analyses = len(results)
    successful = sum(1 for item in results if item.get("status") == "success")
    success_rate = successful / total_analyses * 100 if total_analyses else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("总分析次数", total_analyses)
    col2.metric("成功分析", successful)
    col3.metric("成功率", f"{success_rate:.1f}%")

    st.divider()

    if st.button("🗑️ 清空历史记录"):
        st.session_state.analysis_results = []
        st.rerun()

    for item in reversed(results):
        header = f"📄 {item['filename']} - {item['timestamp']}"
        if item.get("from_cache"):
            header += " [缓存]"
        with st.expander(header, expanded=False):
            st.write(f"**文件名：** {item['filename']}")
            st.write(f"**时间：** {item['timestamp']}")
            if item.get("model_name"):
                st.write(f"**模型：** {item['model_name']}")
            if item.get("image_size"):
                st.write(f"**尺寸：** {item['image_size']}")
            if item.get("processing_time"):
                st.write(f"**处理时间：** {item['processing_time']}")

            if item.get("answer_content"):
                with st.expander("📋 诊断结果", expanded=True):
                    st.write(item["answer_content"])

            if item.get("think_content"):
                with st.expander("🤔 分析过程", expanded=False):
                    st.write(item["think_content"])

            if not item.get("answer_content") and not item.get("think_content"):
                st.write("**诊断结果：**")
                st.write(item.get("raw_result", ""))


def _render_cache_tab() -> None:
    st.header("缓存管理")

    st.markdown(
        """
        ### 🗃️ 数据库缓存系统
        - 🔍 **图片识别**：基于图片内容的 MD5 哈希值
        - 📝 **提示词匹配**：基于提示词内容的 MD5 哈希值
        - 🤖 **模型区分**：不同模型的结果独立缓存
        - ⚡ **快速查询**：SQLite 索引优化检索性能
        """
    )

    try:
        metrics = fetch_cache_metrics()
    except Exception as exc:
        st.error(f"❌ 获取缓存信息失败：{exc}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总缓存记录", metrics["total_records"])
    col2.metric("近7天记录", metrics["recent_records"])
    col3.metric("数据库大小", f"{metrics['db_size_mb']:.2f} MB")
    col4.metric("缓存文件", "medical_analysis_cache.db")

    if metrics["model_stats"]:
        st.subheader("📊 模型使用统计")
        for model_name, count in metrics["model_stats"]:
            st.write(f"**{model_name}**：{count} 次分析")

    st.divider()
    st.subheader("📋 缓存历史记录")

    record_limit = st.selectbox("显示记录数", [10, 25, 50, 100], index=1)
    try:
        history_records = get_analysis_history(record_limit)
    except Exception as exc:
        st.error(f"❌ 获取历史记录失败：{exc}")
        history_records = []

    if history_records:
        for record in history_records:
            header = f"📄 {record['image_name']} - {record['created_at'][:19]}"
            with st.expander(header, expanded=False):
                st.write(f"**文件名：** {record['image_name']}")
                st.write(f"**图片尺寸：** {record['image_size']}")
                st.write(f"**模型：** {record['model_name']} ({record['model_type']})")
                st.write(f"**处理时间：** {record['processing_time']:.2f} 秒")
                st.write(f"**缓存时间：** {record['created_at'][:19]}")

                if record.get("answer_content"):
                    with st.expander("📋 诊断结果", expanded=True):
                        st.write(record["answer_content"])
                if record.get("think_content"):
                    with st.expander("🤔 分析过程", expanded=False):
                        st.write(record["think_content"])
    else:
        st.info("📭 暂无缓存记录")

    st.divider()
    st.subheader("🛠️ 缓存操作")

    col_clear, col_export = st.columns(2)

    with col_clear:
        if st.button("🗑️ 清空所有缓存", type="secondary"):
            try:
                clear_analysis_cache()
            except Exception as exc:
                st.error(f"❌ 清空缓存失败：{exc}")
            else:
                st.success("✅ 缓存已清空")
                st.rerun()

    with col_export:
        if st.button("📤 导出缓存数据"):
            try:
                import pandas as pd

                records = fetch_all_cache_records()
                if not records:
                    st.info("📭 暂无缓存数据可导出")
                else:
                    df = pd.DataFrame(records)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载CSV文件",
                        data=csv_data,
                        file_name=f"analysis_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                    st.success("✅ 导出准备完成")
            except ImportError:
                st.warning("⚠️ 需要安装 pandas 库才能导出 CSV")
            except Exception as exc:
                st.error(f"❌ 导出失败：{exc}")


def main() -> None:
    init_database()
    available_models = build_available_models()
    _ensure_session_state(available_models)
    st.markdown(
        '<h1 class="main-header">🏥 医学图像AI诊断系统</h1>', unsafe_allow_html=True
    )
    _render_sidebar()

    if not st.session_state.available_models:
        st.warning("⚠️ 未检测到可用模型，请检查本地缓存目录或网络配置。")
        return

    if not st.session_state.model_loaded or not st.session_state.current_model:
        st.warning("⚠️ 请先在侧边栏加载模型")
        return

    tab_single, tab_batch, tab_history, tab_cache = st.tabs(
        ["🖼️ 单张图像分析", "📊 批量处理", "📋 历史结果", "🗃️ 缓存管理"]
    )

    with tab_single:
        _render_single_analysis_tab()

    with tab_batch:
        _render_batch_analysis_tab()

    with tab_history:
        _render_history_tab()

    with tab_cache:
        _render_cache_tab()


if __name__ == "__main__":
    main()
