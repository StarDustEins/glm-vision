from __future__ import annotations

import base64
import binascii
import contextlib
import json
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr


from moviepy import VideoFileClip

from app_logic import _call_siliconflow_chat_completion

MODEL_IDENTIFIER = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_VIDEO_SIZE_MB = 80
DEFAULT_AUDIO_SAMPLE_RATE = 16_000

SYSTEM_PROMPT = """
你是专业的多模态语音助手，负责精准转写和结构化整理中文或中英混合对话。
要求：
1. 判断发言人并使用小写字母 a/b/c 依次标注不同说话人，若超过三位则在 c 后继续复用 a/b/c。
2. 标注每句对话的相对时间范围（单位秒，允许小数，起始为 0.0），如缺少置信时间请根据语音节奏估算。
3. 仅输出 JSON，不添加额外解释或自然语言。
4. JSON 结构：
{
  "dialogue": [
    {"speaker": "a", "text": "...", "start": 0.0, "end": 2.6},
    {"speaker": "b", "text": "...", "start": 2.7, "end": 5.4},
    {"speaker": "c", "text": "...", "start": 5.5, "end": 8.8}
  ]
}
5. text 字段需为纯文本，无需标点补偿或舞台指示。英文保持原文。
如果视频无法识别语音，请返回 {"dialogue": []}。
""".strip()

USER_PROMPT = """
请解析上传的视频文件，提取完整的口语对话并按照发言人 a/b/c 区分，返回满足约定 schema 的 JSON。
""".strip()


@dataclass(slots=True)
class VideoSource:
    path: str
    display_name: str
    temporary: bool = False


def _format_status(message: str, level: str = "info") -> str:
    prefix_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    prefix = prefix_map.get(level, "ℹ️")
    return f"{prefix} {message}"


def _resolve_base64_payload(data: str) -> Tuple[bytes, Optional[str]]:
    if not data.startswith("data:"):
        raise ValueError("缺少 data URI 前缀。")
    try:
        header, encoded = data.split(",", 1)
    except ValueError as exc:  # noqa: TRY003
        raise ValueError("data URI 格式不正确。") from exc

    mime_type = None
    if header.startswith("data:"):
        mime_type = header[5:].split(";")[0] or None

    try:
        decoded = base64.b64decode(encoded)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("data URI 解码失败。") from exc
    return decoded, mime_type


def _resolve_video_source(raw_value: Any) -> VideoSource:
    if raw_value is None:
        raise ValueError("尚未提供视频。")

    if isinstance(raw_value, str):
        path = Path(raw_value)
        if not path.exists():
            raise FileNotFoundError("上传的视频文件已不存在，请重新上传。")
        return VideoSource(path=str(path), display_name=path.name)

    if isinstance(raw_value, dict):
        if "path" in raw_value and raw_value["path"]:
            path_value = Path(str(raw_value["path"]))
            if not path_value.exists():
                raise FileNotFoundError("上传的视频临时文件不可用，请重新上传。")
            display_name = str(raw_value.get("name") or path_value.name)
            return VideoSource(
                path=str(path_value),
                display_name=display_name,
            )

        url_field = raw_value.get("url")
        if isinstance(url_field, str):
            if url_field.startswith("file="):
                candidate_path = Path(url_field[5:])
                if candidate_path.exists():
                    display_name = str(raw_value.get("name") or candidate_path.name)
                    return VideoSource(
                        path=str(candidate_path),
                        display_name=display_name,
                    )
            if url_field.startswith("data:"):
                data_payload, mime_type = _resolve_base64_payload(url_field)
                display_name = str(raw_value.get("name") or "uploaded_video")
                suffix = Path(display_name).suffix
                if not suffix and mime_type:
                    suffix = mimetypes.guess_extension(mime_type) or ".mp4"
                elif not suffix:
                    suffix = ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(data_payload)
                    temp_path = tmp.name
                return VideoSource(
                    path=temp_path,
                    display_name=display_name,
                    temporary=True,
                )

        data_field = raw_value.get("data")
        if isinstance(data_field, dict):
            nested_url = data_field.get("url") or data_field.get("path")
            if isinstance(nested_url, str):
                raw_value = dict(raw_value)
                raw_value["url"] = nested_url
                return _resolve_video_source(raw_value)
            nested_data = data_field.get("data")
            if isinstance(nested_data, (str, bytes)):
                data_field = nested_data

        if isinstance(data_field, str):
            payload, mime_type = _resolve_base64_payload(data_field)
            display_name = str(raw_value.get("name") or "uploaded_video")
        elif isinstance(data_field, bytes):
            payload = data_field
            mime_type = str(raw_value.get("mime_type") or "") or None
            display_name = str(raw_value.get("name") or "uploaded_video")
        else:
            raise ValueError("无法解析上传的视频内容。")

        suffix = Path(display_name).suffix
        if not suffix and mime_type:
            suffix = mimetypes.guess_extension(mime_type) or ".mp4"
        elif not suffix:
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            temp_path = tmp.name

        return VideoSource(
            path=temp_path,
            display_name=display_name,
            temporary=True,
        )

    raise ValueError("不支持的上传结果类型。")


def _extract_audio_to_base64(
    video_path: str, *, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> Tuple[str, str]:

    path = Path(video_path or "")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("视频文件不存在，请重新上传。")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        raise ValueError(
            f"视频文件过大（{file_size_mb:.1f} MB），请控制在 {MAX_VIDEO_SIZE_MB} MB 以内。"
        )

    with VideoFileClip(str(path)) as clip:
        if clip.audio is None:
            raise ValueError("视频缺少可用的音频轨道，无法执行语音解析。")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio_path = tmp.name

        try:
            clip.audio.write_audiofile(
                temp_audio_path,
                codec="pcm_s16le",
                fps=sample_rate,
                # verbose=False,
                logger=None,
            )

            with open(temp_audio_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_audio_path)

    return f"data:audio/wav;base64,{encoded_audio}", "wav"


def _extract_tag_content(text: str, tag: str) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    start_marker = f"<{tag}>"
    end_marker = f"</{tag}>"
    start_idx = lower.find(start_marker)
    end_idx = lower.find(end_marker)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
    start = start_idx + len(start_marker)
    return text[start:end_idx].strip()


def _parse_answer_json(answer_segment: str) -> Dict[str, Any]:
    if not answer_segment:
        raise ValueError("模型未返回可用内容。")

    candidate = answer_segment.strip()
    if not (candidate.startswith("{") and candidate.endswith("}")):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效的 JSON 片段。")
        candidate = candidate[start : end + 1]

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型返回的 JSON 解析失败：{exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("模型返回内容不是 JSON 对象。")

    dialogue = payload.get("dialogue")
    if dialogue is None:
        raise ValueError("JSON 中缺少 dialogue 字段。")
    if not isinstance(dialogue, list):
        raise ValueError("dialogue 字段必须为列表。")

    normalized: List[Dict[str, Any]] = []
    for item in dialogue:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", "")).strip().lower()
        if speaker not in {"a", "b", "c"}:
            continue
        text_value = str(item.get("text", "")).strip()
        if not text_value:
            continue
        entry: Dict[str, Any] = {"speaker": speaker, "text": text_value}
        if "start" in item:
            try:
                entry["start"] = float(item["start"])
            except (TypeError, ValueError):
                pass
        if "end" in item:
            try:
                entry["end"] = float(item["end"])
            except (TypeError, ValueError):
                pass
        normalized.append(entry)

    payload["dialogue"] = normalized
    return payload


def _request_dialogue_payload(source: VideoSource) -> Tuple[Dict[str, Any], str]:
    audio_base64, audio_format = _extract_audio_to_base64(source.path)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": audio_base64,
                    },
                },
            ],
        },
    ]

    content_text, reasoning_text = _call_siliconflow_chat_completion(
        messages,
        model_identifier=MODEL_IDENTIFIER,
        extra_options={
            "modalities": ["text", "audio"],
            "audio": {"voice": "Cherry", "format": audio_format},
        },
    )

    answer_segment = (
        _extract_tag_content(content_text, "answer") or content_text
    ).strip()
    think_segment = (
        _extract_tag_content(content_text, "think") or reasoning_text or ""
    ).strip()

    payload = _parse_answer_json(answer_segment)
    return payload, think_segment


def _handle_analysis(video_input: Any) -> Tuple[str, Any, str, str]:
    if not video_input:
        return (
            _format_status("请先上传视频文件。", "warning"),
            gr.update(value=None),
            "尚未生成 JSON 输出。",
            "暂无推理内容。",
        )

    try:
        source = _resolve_video_source(video_input)
    except Exception as exc:  # noqa: BLE001
        return (
            _format_status(f"视频处理失败：{exc}", "error"),
            gr.update(value=None),
            "尚未生成 JSON 输出。",
            f"```text\n{exc}\n```",
        )

    try:
        payload, think_segment = _request_dialogue_payload(source)
    except Exception as exc:  # noqa: BLE001
        return (
            _format_status(f"解析失败：{exc}", "error"),
            gr.update(value=None),
            "尚未生成 JSON 输出。",
            f"```text\n{exc}\n```",
        )
    finally:
        if source.temporary:
            with contextlib.suppress(OSError):
                os.remove(source.path)

    prettified = json.dumps(payload, ensure_ascii=False, indent=2)
    answer_md = f"```json\n{prettified}\n```"
    think_display = (
        f"```text\n{think_segment}\n```"
        if think_segment
        else "```text\n（模型未返回显式推理过程）\n```"
    )

    return (
        _format_status("解析完成，以下为结构化对话结果。", "success"),
        payload,
        answer_md,
        think_display,
    )


def main() -> gr.Blocks:
    with gr.Blocks(
        title="视频对话解析",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 视频对话解析")
        gr.Markdown(
            "上传一个包含对话的视频，模型进行语音识别与角色区分，输出结构化 JSON。"
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="上传并预览视频",
                    sources=["upload"],
                )
            with gr.Column(scale=1):
                analyze_button = gr.Button("解析视频", variant="primary")
                status_display = gr.Markdown(
                    _format_status("等待上传视频文件。", "info"),
                )
                gr.Markdown("### 解析结果")
                dialogue_json = gr.JSON(label="对话 JSON")
                # answer_md = gr.Markdown("尚未生成 JSON 输出。")
                # gr.Markdown("### 推理过程")
                # think_md = gr.Markdown("暂无推理内容。")

        analyze_button.click(
            fn=_handle_analysis,
            inputs=video_input,
            outputs=[status_display, dialogue_json],
        )

    return demo


if __name__ == "__main__":
    app = main()
    app.launch(
        pwa=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7861)),
    )
