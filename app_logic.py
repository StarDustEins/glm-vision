from __future__ import annotations

import base64
import hashlib
import io
import os
import sqlite3
import time
from collections import OrderedDict
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from urllib import error as urllib_error
from urllib import request as urllib_request
import torch
from PIL import Image
from modelscope import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
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

_LOADED_MODELS: Dict[str, Dict[str, Any]] = {}

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

DEFAULT_MEDICAL_PROMPT = """
你是资深影像科医生兼医学影像审核员，请用简体中文。

<think>
逐步说明你的推理：先判断图像是否医学影像（如X光、CT、MRI、超声、病理切片等），再分析主要征象、诊断逻辑、置信度依据，并记录任何不确定性或补充需求。
</think>

若图像并非医学影像或信息不足，请在 <answer> 中仅返回：
温馨提示：当前上传的图像似乎不是医学影像，请检查后重新上传。（可附一句原因或所需资料）

若确认是医学影像，请在 <answer> 中按以下格式输出，保持语句简洁：
1. 部位信息：…
2. 检查所见：…
3. 诊断意见：…
4. 置信度：高/中/低（简述依据或补充需求）

最终输出格式必须严格为：
<think>你的思考过程</think>
<answer>你的最终回答</answer>
"""

SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-khdvgcgwzbhwqhfcrlbaoazfegxfypunnlgpnwkhbydqqvth",
)
try:
    SILICONFLOW_TIMEOUT = float(os.environ.get("SILICONFLOW_TIMEOUT", "60"))
except ValueError:
    SILICONFLOW_TIMEOUT = 60.0


def _is_siliconflow_model(model: object) -> bool:
    return isinstance(model, dict) and model.get("inference_mode") == "siliconflow"

REMOTE_MODEL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "Qwen3-VL-8B-Instruct": {
        "identifier": "Qwen/Qwen3-VL-8B-Instruct",
        "description": "Qwen3 VL 8B 视觉语言模型（SiliconFlow 云端）",
        "model_type": "qwen3_vl",
        "inference_mode": "siliconflow",
    },
    "Qwen3-VL-8B-Thinking": {
        "identifier": "Qwen/Qwen3-VL-8B-Thinking",
        "description": "Qwen3 VL 8B 视觉语言模型（SiliconFlow 云端）",
        "model_type": "qwen3_vl",
        "inference_mode": "siliconflow",
    },
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
}

SUPPORTED_MODEL_TYPES = {"glm4v", "medgemma", "qwen3_vl"}


def _encode_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def _collect_message_text(payload: object) -> str:
    """将SiliconFlow返回的消息片段统一拼接为字符串。"""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        if "text" in payload:
            value = payload.get("text")
            return value if isinstance(value, str) else _collect_message_text(value)
        if "content" in payload:
            return _collect_message_text(payload.get("content"))
    if isinstance(payload, list):
        parts: List[str] = []
        for item in payload:
            text = _collect_message_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


def _call_siliconflow_chat_completion(
    messages: List[Dict[str, object]],
    *,
    model_identifier: str,
    base_url: Optional[str] = None,
) -> Tuple[str, str]:
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("未配置 SiliconFlow API Key，无法调用云端算力")

    endpoint_base = (base_url or SILICONFLOW_BASE_URL).rstrip("/")
    request_url = f"{endpoint_base}/chat/completions"
    payload = {
        "model": model_identifier,
        "messages": messages,
        "stream": False,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    request = urllib_request.Request(request_url, data=data, headers=headers, method="POST")

    try:
        with urllib_request.urlopen(request, timeout=SILICONFLOW_TIMEOUT) as response:
            raw_body = response.read()
    except urllib_error.HTTPError as exc:
        detail_bytes = exc.read() or b""
        detail_message = exc.reason
        if detail_bytes:
            try:
                detail_payload = json.loads(detail_bytes.decode("utf-8"))
                if isinstance(detail_payload, dict):
                    detail_message = detail_payload.get("error") or detail_payload
            except (UnicodeDecodeError, json.JSONDecodeError):
                detail_message = detail_bytes.decode("utf-8", errors="ignore") or detail_message
        raise RuntimeError(f"SiliconFlow API 请求失败：{detail_message}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"无法连接 SiliconFlow API：{exc}") from exc

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("SiliconFlow API 响应解析失败") from exc

    error_info = body.get("error")
    if error_info:
        if isinstance(error_info, dict):
            message = error_info.get("message") or json.dumps(error_info, ensure_ascii=False)
        else:
            message = str(error_info)
        raise RuntimeError(f"SiliconFlow API 返回错误：{message}")

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("SiliconFlow API 返回空响应")

    message = choices[0].get("message", {})
    content_text = _collect_message_text(message.get("content"))
    reasoning_payload = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
    )
    reasoning_text = _collect_message_text(reasoning_payload)

    if not isinstance(content_text, str):
        raise RuntimeError("SiliconFlow API 返回的消息格式无法解析")

    return content_text.strip(), reasoning_text.strip()


def _resolve_vllm_gpu_utilization(config: Dict[str, Any]) -> float:
    env_value = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    config_value = config.get("gpu_memory_utilization", 0.7)
    try:
        return float(env_value if env_value is not None else config_value)
    except (TypeError, ValueError):
        return 0.7


def _resolve_vllm_max_model_len(config: Dict[str, Any]) -> int:
    env_value = os.environ.get("VLLM_MAX_MODEL_LEN")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass

    config_value = config.get("max_model_len")
    if config_value is not None:
        try:
            return int(config_value)
        except (TypeError, ValueError):
            pass

    return 32768


def _resolve_sampling_params(model_type: str, config: Dict[str, Any]) -> SamplingParams:
    default_max_tokens = {
        "glm4v": 2048,
        "medgemma": 1024,
        "qwen3_vl": 1024,
    }.get(model_type, 1024)

    max_tokens = config.get("max_tokens", default_max_tokens)
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = default_max_tokens

    temperature = config.get("temperature", 0.7 if model_type != "glm4v" else 0.8)
    top_p = config.get("top_p", 0.9)

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7 if model_type != "glm4v" else 0.8

    try:
        top_p = float(top_p)
    except (TypeError, ValueError):
        top_p = 0.9

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=[],
    )


def _prepare_inputs_for_vllm(messages: List[Dict[str, object]], processor) -> Dict[str, object]:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_patch_size = getattr(
        getattr(processor, "image_processor", None), "patch_size", None
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=image_patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    multimodal_data: Dict[str, object] = {}
    if image_inputs is not None:
        multimodal_data["image"] = image_inputs
    if video_inputs is not None:
        multimodal_data["video"] = video_inputs

    return {
        "prompt": prompt,
        "multi_modal_data": multimodal_data,
        "mm_processor_kwargs": video_kwargs,
    }


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
            config["source"] = "remote"
            config["location"] = identifier
            if config.get("inference_mode") == "siliconflow":
                config["path"] = "SiliconFlow Cloud"
                config.setdefault("base_url", SILICONFLOW_BASE_URL)
            else:
                config["path"] = identifier

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
    """加载指定模型，返回 (model_bundle, processor, device, model_type)。"""
    model_type = model_config.get("model_type", "glm4v")
    identifier = model_config.get("identifier", model_name)
    inference_mode = model_config.get("inference_mode")

    if inference_mode == "siliconflow":
        location = f"siliconflow::{identifier}"
        cached = _LOADED_MODELS.get(model_name)
        if cached and cached.get("location") == location:
            return (
                cached["model"],
                cached["processor"],
                cached["device"],
                cached["model_type"],
            )

        base_url = model_config.get("base_url") or SILICONFLOW_BASE_URL
        model_bundle = {
            "inference_mode": "siliconflow",
            "identifier": identifier,
            "base_url": base_url,
        }

        _LOADED_MODELS[model_name] = {
            "model": model_bundle,
            "processor": None,
            "device": "siliconflow",
            "model_type": model_type,
            "location": location,
        }

        return model_bundle, None, "siliconflow", model_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    location = model_config.get("path") or identifier

    cached = _LOADED_MODELS.get(model_name)
    if cached and cached.get("location") == location:
        return (
            cached["model"],
            cached["processor"],
            cached["device"],
            cached["model_type"],
        )

    model_path = location
    if not os.path.exists(model_path):
        model_path = identifier

    tensor_parallel = max(torch.cuda.device_count(), 1)
    gpu_memory_util = _resolve_vllm_gpu_utilization(model_config)
    max_model_len = _resolve_vllm_max_model_len(model_config)
    sampling_params = _resolve_sampling_params(model_type, model_config)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_memory_util,
        max_model_len=max_model_len,
        seed=0,
    )

    if model_type == "medgemma":
        processor = MedGemmaProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        except TypeError:
            processor = AutoProcessor.from_pretrained(model_path)

    model_bundle = {"llm": llm, "sampling_params": sampling_params}

    _LOADED_MODELS[model_name] = {
        "model": model_bundle,
        "processor": processor,
        "device": device,
        "model_type": model_type,
        "location": location,
    }

    return model_bundle, processor, device, model_type


def get_loaded_model_state(
    model_name: Optional[str] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """返回已缓存的模型状态，用于跨会话同步。"""
    if model_name:
        cached = _LOADED_MODELS.get(model_name)
        if cached is None:
            return None
        return model_name, cached

    if not _LOADED_MODELS:
        return None

    name, cached = next(iter(_LOADED_MODELS.items()))
    return name, cached


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


def process_image_analysis(
    image: Image.Image,
    model,
    processor,
    device: Any,
    model_type: str,
    model_name: str,
    *,
    image_name: str,
    prompt: str = DEFAULT_MEDICAL_PROMPT,
    enable_cache: bool = True,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    """执行图像分析，返回分析结果。"""
    remote_inference = _is_siliconflow_model(model)
    if model is None or (processor is None and not remote_inference):
        raise ValueError("模型尚未加载")

    if enable_cache:
        cached = get_cached_result(image, model_type, model_name, prompt)
        if cached:
            return cached

    start_time = time.time()

    reasoning_text = ""
    content_text = ""

    if remote_inference:
        image_payload = _encode_image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_payload, "detail": "high"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        model_identifier = model.get("identifier", model_name)
        base_url = model.get("base_url")
        content_text, reasoning_text = _call_siliconflow_chat_completion(
            messages,
            model_identifier=model_identifier,
            base_url=base_url,
        )

        raw_text = content_text.strip()
        think_lower = "<think>" in raw_text.lower()
        answer_lower = "<answer>" in raw_text.lower()
        if reasoning_text and not think_lower:
            composed_think = f"<think>{reasoning_text.strip()}</think>"
            if answer_lower:
                raw_text = f"{composed_think}\n{raw_text}"
            else:
                raw_text = f"{composed_think}\n<answer>{raw_text}</answer>"

        if stream_callback and raw_text:
            stream_callback(raw_text)
    else:
        image_input = (
            _encode_image_to_base64(image) if model_type == "qwen3_vl" else image
        )
        multimodal_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        vllm_inputs = _prepare_inputs_for_vllm(multimodal_messages, processor)
        model_bundle = model if isinstance(model, dict) else {"llm": model}
        llm = model_bundle["llm"]
        sampling_params = model_bundle.get("sampling_params") or _resolve_sampling_params(
            model_type, {}
        )

        outputs = llm.generate([vllm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text or ""

        if stream_callback and generated_text:
            stream_callback(generated_text)

        raw_text = generated_text.strip()
        content_text = raw_text

    elapsed = time.time() - start_time
    think_content, answer_content = parse_analysis_result(raw_text)

    if not think_content and reasoning_text:
        think_content = reasoning_text.strip()
    if not answer_content:
        fallback = content_text.strip() if content_text else raw_text
        answer_content = fallback

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
    "get_loaded_model_state",
    "init_database",
    "load_model",
    "parse_analysis_result",
    "process_image_analysis",
    "save_analysis_result",
    "validate_image",
]
