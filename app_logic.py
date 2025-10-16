from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import threading
import time
from collections import OrderedDict
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
import torch
from PIL import Image
from modelscope import (
    AutoProcessor,
    Glm4vForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from transformers import AutoProcessor as MedGemmaProcessor

SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
DATABASE_PATH = "medical_analysis_cache.db"
DEFAULT_MODEL_BASE = (
    r"\\wsl.localhost\Ubuntu-24.04\home\elysion\.cache\modelscope\hub\models"
)
DEFAULT_MODEL_BASES = [
    DEFAULT_MODEL_BASE,
    os.path.expanduser("~/.cache/modelscope/hub/models"),
]

DEFAULT_MEDICAL_PROMPT = """
你是资深影像科医生。结合所给医学图像，请直接输出以下三部分内容，保持语言精炼、专业、中文：

1. **部位判断**：明确图像对应的解剖部位或器官。
2. **检查所见**：概述主要影像表现，可包含异常与否、形态、密度/信号特点等。
3. **诊断意见**：给出最可能诊断及必要的鉴别或进一步检查建议。

若信息不足，请说明“不足以判断”并提示需要的补充资料。
"""

REMOTE_MODEL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "GLM-4V-9B": {
        "identifier": "ZhipuAI/GLM-4.1V-9B-Thinking",
        "description": "GLM-4V多模态大模型 (9B参数)",
        "model_type": "glm4v",
    },
    "MedGemma-4B": {
        "identifier": "google/medgemma-4b-it",
        "description": "MedGemma医学专用模型 (4B参数)",
        "model_type": "medgemma",
    },
    "Qwen3-VL-8B": {
        "identifier": "Qwen/Qwen3-VL-8B-Instruct",
        "description": "Qwen3 VL 8B 视觉语言模型",
        "model_type": "qwen3_vl",
    },
}

SUPPORTED_MODEL_TYPES = {"glm4v", "medgemma", "qwen3_vl"}


@dataclass(slots=True)
class AnalysisResult:
    raw_result: str
    think_content: str
    answer_content: str
    processing_time: float
    created_at: Optional[str]
    from_cache: bool


def _resolve_local_model_path(identifier: str) -> Optional[str]:
    """尝试根据模型标识符定位本地缓存目录。"""
    normalized = identifier.replace("\\", "/").split("/")
    for base in DEFAULT_MODEL_BASES:
        candidate_path = Path(os.path.join(base, *normalized))
        if candidate_path.is_dir() and _contains_files(candidate_path):
            return str(candidate_path)
    return None


def _contains_files(directory: Path) -> bool:
    """判断目录下是否存在至少一个文件（递归遍历）。"""
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            for entry in current.iterdir():
                if entry.is_file():
                    return True
                if entry.is_dir():
                    stack.append(entry)
        except (PermissionError, OSError):
            continue
    return False


def _discover_local_catalog(base_path: str) -> "OrderedDict[str, Dict[str, str]]":
    """从指定目录中发现本地模型缓存。"""
    results: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    if not os.path.isdir(base_path):
        return results

    base = Path(base_path)
    try:
        owners = sorted(
            [entry for entry in base.iterdir() if entry.is_dir()],
            key=lambda p: p.name.lower(),
        )
    except PermissionError:
        return results

    for owner in owners:
        try:
            models = sorted(
                [
                    entry
                    for entry in owner.iterdir()
                    if entry.is_dir() and _contains_files(entry)
                ],
                key=lambda p: p.name.lower(),
            )
        except PermissionError:
            continue

        for model_dir in models:
            config_path = model_dir / "config.json"
            model_type = "glm4v"
            description = f"{owner.name}/{model_dir.name}"

            if config_path.is_file():
                try:
                    with config_path.open("r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    model_type = cfg.get("model_type", model_type)
                    if cfg.get("model_name"):
                        description = cfg["model_name"]
                except (json.JSONDecodeError, OSError):
                    pass

            lower_name = model_dir.name.lower()
            if "qwen" in lower_name or "qwen" in owner.name.lower():
                model_type = "qwen3_vl"

            key = f"{owner.name}/{model_dir.name}"
            if key in results or model_type not in SUPPORTED_MODEL_TYPES:
                continue

            results[key] = {
                "identifier": key,
                "description": description,
                "model_type": model_type,
                "path": str(model_dir),
                "location": str(model_dir),
                "source": "local",
            }

    return results


def build_available_models() -> Dict[str, Dict[str, str]]:
    """构建可用模型列表，优先使用本地缓存。"""
    models: "OrderedDict[str, Dict[str, str]]" = OrderedDict()

    for name, template in REMOTE_MODEL_TEMPLATES.items():
        config = dict(template)
        identifier = config["identifier"]
        local_path = _resolve_local_model_path(identifier)

        if local_path:
            config["path"] = local_path
            config["source"] = "local"
            config["location"] = local_path
        else:
            config["path"] = identifier
            config["source"] = "remote"
            config["location"] = identifier

        if config.get("model_type") in SUPPORTED_MODEL_TYPES:
            models[name] = config

    for base_path in DEFAULT_MODEL_BASES:
        for name, config in _discover_local_catalog(base_path).items():
            if name not in models and config.get("model_type") in SUPPORTED_MODEL_TYPES:
                models[name] = config

    return models


def init_database() -> None:
    """初始化SQLite数据库。"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE NOT NULL,
                image_name TEXT NOT NULL,
                image_size TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                raw_result TEXT NOT NULL,
                think_content TEXT,
                answer_content TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_image_hash ON analysis_results(image_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON analysis_results(created_at)"
        )
        conn.commit()
    finally:
        conn.close()


def _calculate_image_hash(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()


def _calculate_prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def save_analysis_result(
    image: Image.Image,
    image_name: str,
    model_type: str,
    model_name: str,
    prompt: str,
    raw_result: str,
    think_content: str,
    answer_content: str,
    processing_time: float,
    created_at: Optional[str] = None,
) -> str:
    """保存分析结果并返回时间戳。出现异常时抛出RuntimeError。"""
    image_hash = _calculate_image_hash(image)
    prompt_hash = _calculate_prompt_hash(prompt)
    image_size = f"{image.size[0]}x{image.size[1]}"
    timestamp = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO analysis_results
            (image_hash, image_name, image_size, model_type, model_name, prompt_hash,
             raw_result, think_content, answer_content, processing_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                image_hash,
                image_name,
                image_size,
                model_type,
                model_name,
                prompt_hash,
                raw_result,
                think_content,
                answer_content,
                processing_time,
                timestamp,
            ),
        )
        conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(f"保存分析结果失败：{exc}") from exc
    finally:
        conn.close()

    return timestamp


def get_cached_result(
    image: Image.Image, model_type: str, model_name: str, prompt: str
) -> Optional[AnalysisResult]:
    """获取缓存结果。若查询失败抛出RuntimeError。"""
    image_hash = _calculate_image_hash(image)
    prompt_hash = _calculate_prompt_hash(prompt)

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT raw_result, think_content, answer_content, processing_time, created_at
            FROM analysis_results
            WHERE image_hash = ? AND model_type = ? AND model_name = ? AND prompt_hash = ?
            ORDER BY created_at DESC LIMIT 1
        """,
            (image_hash, model_type, model_name, prompt_hash),
        )
        row = cursor.fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(f"查询缓存失败：{exc}") from exc
    finally:
        conn.close()

    if not row:
        return None

    raw_result, think_content, answer_content, processing_time, created_at = row
    return AnalysisResult(
        raw_result=raw_result or "",
        think_content=think_content or "",
        answer_content=answer_content or "",
        processing_time=float(processing_time or 0.0),
        created_at=created_at,
        from_cache=True,
    )


def get_analysis_history(limit: int = 50) -> List[Dict[str, str]]:
    """返回指定数量的历史记录。"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT image_name, image_size, model_type, model_name,
                   think_content, answer_content, processing_time, created_at
            FROM analysis_results
            ORDER BY created_at DESC LIMIT ?
        """,
            (limit,),
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"获取历史记录失败：{exc}") from exc
    finally:
        conn.close()

    history: List[Dict[str, str]] = []
    for row in rows:
        history.append(
            {
                "image_name": row[0],
                "image_size": row[1],
                "model_type": row[2],
                "model_name": row[3],
                "think_content": row[4] or "",
                "answer_content": row[5] or "",
                "processing_time": float(row[6] or 0.0),
                "created_at": row[7],
            }
        )
    return history


def fetch_cache_metrics() -> Dict[str, object]:
    """获取缓存统计信息。"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_results")
        total_records = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT model_name, COUNT(*)
            FROM analysis_results
            GROUP BY model_name
        """
        )
        model_stats = cursor.fetchall()

        cursor.execute(
            """
            SELECT COUNT(*) FROM analysis_results
            WHERE created_at >= datetime('now', '-7 days')
        """
        )
        recent_records = cursor.fetchone()[0]

        cursor.execute(
            "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
        )
        size_row = cursor.fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(f"获取缓存信息失败：{exc}") from exc
    finally:
        conn.close()

    db_size = 0.0
    if size_row and size_row[0] is not None:
        db_size = float(size_row[0]) / (1024 * 1024)

    return {
        "total_records": total_records,
        "recent_records": recent_records,
        "model_stats": model_stats,
        "db_size_mb": db_size,
    }


def fetch_all_cache_records() -> List[Dict[str, object]]:
    """导出所有缓存记录。"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM analysis_results ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    except sqlite3.Error as exc:
        raise RuntimeError(f"导出缓存数据失败：{exc}") from exc
    finally:
        conn.close()

    return [dict(zip(columns, row)) for row in rows]


def clear_analysis_cache() -> None:
    """清空缓存。"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_results")
        conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(f"清空缓存失败：{exc}") from exc
    finally:
        conn.close()


def load_model(model_name: str, model_config: Dict[str, str]):
    """加载指定模型，返回 (model, processor, device, model_type)。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_config["path"]
    model_type = model_config.get("model_type", "glm4v")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if model_type == "glm4v":
        model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            dtype=dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        return model, processor, device, "glm4v"

    if model_type == "medgemma":
        model_load_path = model_path
        if not os.path.exists(model_load_path):
            model_load_path = model_config.get("identifier", model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            dtype=dtype,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        processor = MedGemmaProcessor.from_pretrained(
            model_load_path, trust_remote_code=True
        )
        return model, processor, device, "medgemma"

    if model_type == "qwen3_vl":
        model_load_path = model_path
        if not os.path.exists(model_load_path):
            model_load_path = model_config.get("identifier", model_path)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_load_path,
            dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_load_path)
        return model, processor, device, "qwen3_vl"

    raise ValueError(f"暂不支持的模型类型：{model_type}")


def parse_analysis_result(raw_text: str) -> tuple[str, str]:
    """解析模型输出，分离think与answer部分。"""
    import re

    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_text, re.DOTALL)

    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else ""

    if not think_content and not answer_content:
        answer_content = raw_text.strip()

    return think_content, answer_content


def _stream_model_output(
    model,
    processor,
    inputs,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stream_callback: Optional[Callable[[str], None]],
) -> str:
    """执行流式生成，并通过回调返回增量文本。"""
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=False
    )

    model_inputs = dict(inputs)
    generation_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": top_p,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    collected_chunks: List[str] = []
    for chunk in streamer:
        if stream_callback:
            stream_callback(chunk)
        collected_chunks.append(chunk)

    thread.join()
    return "".join(collected_chunks).strip()


def process_image_analysis(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    model_type: str,
    model_name: str,
    *,
    image_name: str,
    prompt: str = DEFAULT_MEDICAL_PROMPT,
    enable_cache: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    """执行图像分析，返回分析结果。"""
    if model is None or processor is None:
        raise ValueError("模型尚未加载")

    if enable_cache:
        cached = get_cached_result(image, model_type, model_name, prompt)
        if cached:
            return cached

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if not hasattr(model, "hf_device_map"):
        inputs = inputs.to(device)

    start_time = time.time()

    if model_type == "glm4v":
        raw_text = _stream_model_output(
            model,
            processor,
            inputs,
            max_new_tokens=8192,
            temperature=0.8,
            top_p=0.9,
            stream_callback=stream_callback,
        )
    elif model_type == "medgemma":
        raw_text = _stream_model_output(
            model,
            processor,
            inputs,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            stream_callback=stream_callback,
        )
    elif model_type == "qwen3_vl":
        raw_text = _stream_model_output(
            model,
            processor,
            inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stream_callback=stream_callback,
        )
    else:
        raise ValueError(f"未知的模型类型：{model_type}")

    elapsed = time.time() - start_time
    think_content, answer_content = parse_analysis_result(raw_text)

    if enable_cache and raw_text:
        created_at = save_analysis_result(
            image=image,
            image_name=image_name,
            model_type=model_type,
            model_name=model_name,
            prompt=prompt,
            raw_result=raw_text,
            think_content=think_content,
            answer_content=answer_content,
            processing_time=elapsed,
        )
    else:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return AnalysisResult(
        raw_result=raw_text,
        think_content=think_content,
        answer_content=answer_content,
        processing_time=elapsed,
        created_at=created_at,
        from_cache=False,
    )


def validate_image(uploaded_file) -> tuple[bool, str]:
    """验证上传的图像文件。"""
    if uploaded_file is None:
        return False, "未选择文件"

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"不支持的格式 {file_ext}。支持的格式：{', '.join(SUPPORTED_FORMATS)}"

    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "文件大小超过10MB限制"

    return True, "文件验证通过"


def get_device_info() -> List[Dict[str, str]]:
    """获取GPU/CUDA核心信息，用于在UI中展示。"""
    info: List[Dict[str, str]] = []

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / (1024**3)
        info.append(
            {
                "label": "GPU",
                "value": f"{gpu_name}",
                "help": f"显存 {total_memory:.1f} GB",
            }
        )
        cuda_version = torch.version.cuda or "未知"
        info.append(
            {
                "label": "CUDA",
                "value": cuda_version,
                "help": "Torch 检测到的 CUDA 版本",
            }
        )
    else:
        info.append(
            {
                "label": "GPU",
                "value": "未检测到 CUDA 设备",
                "help": "模型将以 CPU 模式运行",
            }
        )

    return info


__all__ = [
    "AnalysisResult",
    "DATABASE_PATH",
    "DEFAULT_MEDICAL_PROMPT",
    "DEFAULT_MODEL_BASE",
    "SUPPORTED_FORMATS",
    "build_available_models",
    "clear_analysis_cache",
    "fetch_all_cache_records",
    "fetch_cache_metrics",
    "get_analysis_history",
    "get_cached_result",
    "get_device_info",
    "init_database",
    "load_model",
    "parse_analysis_result",
    "process_image_analysis",
    "save_analysis_result",
    "validate_image",
]
