import streamlit as st
import torch
import time
import os
from PIL import Image
from modelscope import AutoProcessor, Glm4vForConditionalGeneration
from transformers import TextIteratorStreamer
import threading
from datetime import datetime
import sqlite3
import hashlib
import re

# 页面配置
st.set_page_config(
    page_title="医学图像AI诊断系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 支持的图片格式
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# 数据库配置
DATABASE_PATH = "medical_analysis_cache.db"

# 默认专业医学诊断提示词
DEFAULT_MEDICAL_PROMPT = """
作为资深医学影像诊断专家，请对这张医学图像进行全面、系统的专业分析。

**分析框架：**

<think>
请按以下步骤进行系统性分析思考：

1. **技术质量评估**
   - 图像采集技术参数（扫描方式、层厚、对比剂等）
   - 图像质量评价（清晰度、伪影、体位等）
   - 解剖结构显示完整性

2. **解剖结构系统评估**
   - 骨骼系统：骨质密度、骨皮质连续性、关节间隙
   - 软组织：肌肉、脂肪层次、筋膜平面
   - 器官形态：大小、形状、位置、密度/信号
   - 血管系统：主要血管走行、管腔情况

3. **异常发现详细分析**
   - 病变位置：精确解剖定位
   - 形态特征：大小、形状、边界、内部结构
   - 密度/信号特征：CT值、T1/T2信号特点
   - 强化模式：对比剂强化特征（如适用）
   - 周围组织关系：浸润、压迫、移位

4. **影像学测量**
   - 病变最大径线测量
   - 多平面测量数据
   - 体积评估（如需要）

5. **鉴别诊断思路**
   - 主要诊断考虑及依据
   - 需要排除的疾病
   - 诊断置信度评估
</think>

<answer>
**影像学报告**

**技术参数：**
[描述扫描技术参数和图像质量]

**影像学所见：**
[按解剖系统系统性描述正常和异常发现]

**测量数据：**
[提供精确的测量结果]

**影像学印象：**
1. 主要诊断考虑
2. 鉴别诊断
3. 建议进一步检查
4. 随访建议

**备注：**
[重要提醒和注意事项]
</answer>

请严格按照上述格式使用中文输出，确保分析的专业性和系统性。
"""

# 模型配置
AVAILABLE_MODELS = {
    "GLM-4V-9B": {
        "path": "ZhipuAI/GLM-4.1V-9B-Thinking",
        "description": "GLM-4V多模态大模型 (9B参数)",
        "type": "multimodal",
    },
    "MedGemma-4B": {
        "path": r"C:\Users\Hi\.cache\modelscope\hub\models\google\medgemma-4b-it",
        "description": "MedGemma医学专用模型 (4B参数)",
        "type": "medical",
    },
}

# 初始化session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "device" not in st.session_state:
    st.session_state.device = None
if "current_model" not in st.session_state:
    st.session_state.current_model = "GLM-4V-9B"
if "model_type" not in st.session_state:
    st.session_state.model_type = "glm4v"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []


# 数据库操作函数
def init_database():
    """初始化SQLite数据库"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # 创建分析结果表
    cursor.execute("""
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
    """)

    # 创建索引以提高查询性能
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_hash ON analysis_results(image_hash)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at ON analysis_results(created_at)
    """)

    conn.commit()
    conn.close()


def calculate_image_hash(image):
    """计算图片的MD5哈希值"""
    # 将PIL图像转换为字节
    import io

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # 计算MD5哈希
    return hashlib.md5(img_byte_arr).hexdigest()


def calculate_prompt_hash(prompt):
    """计算提示词的MD5哈希值"""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def save_analysis_result(
    image,
    image_name,
    model_type,
    model_name,
    prompt,
    raw_result,
    think_content,
    answer_content,
    processing_time,
):
    """保存分析结果到数据库"""
    try:
        image_hash = calculate_image_hash(image)
        prompt_hash = calculate_prompt_hash(prompt)
        image_size = f"{image.size[0]}x{image.size[1]}"

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO analysis_results 
            (image_hash, image_name, image_size, model_type, model_name, prompt_hash,
             raw_result, think_content, answer_content, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"保存分析结果失败：{e}")
        return False


def get_cached_result(image, model_type, model_name, prompt):
    """从数据库获取缓存的分析结果"""
    try:
        image_hash = calculate_image_hash(image)
        prompt_hash = calculate_prompt_hash(prompt)

        conn = sqlite3.connect(DATABASE_PATH)
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

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "raw_result": result[0],
                "think_content": result[1],
                "answer_content": result[2],
                "processing_time": result[3],
                "created_at": result[4],
                "from_cache": True,
            }
        return None
    except Exception as e:
        st.error(f"查询缓存失败：{e}")
        return None


def get_analysis_history(limit=50):
    """获取分析历史记录"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
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

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "image_name": row[0],
                "image_size": row[1],
                "model_type": row[2],
                "model_name": row[3],
                "think_content": row[4],
                "answer_content": row[5],
                "processing_time": row[6],
                "created_at": row[7],
            }
            for row in results
        ]
    except Exception as e:
        st.error(f"获取历史记录失败：{e}")
        return []


def clear_analysis_cache():
    """清空分析缓存"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_results")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"清空缓存失败：{e}")
        return False


def update_processing_time(image, model_type, model_name, prompt, processing_time):
    """更新数据库中的处理时间"""
    try:
        image_hash = calculate_image_hash(image)
        prompt_hash = calculate_prompt_hash(prompt)

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE analysis_results 
            SET processing_time = ?
            WHERE image_hash = ? AND model_type = ? AND model_name = ? AND prompt_hash = ?
        """,
            (processing_time, image_hash, model_type, model_name, prompt_hash),
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"更新处理时间失败：{e}")
        return False


# 初始化数据库
init_database()


@st.cache_resource
def load_model(model_name):
    """加载指定的模型（缓存以避免重复加载）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name not in AVAILABLE_MODELS:
        st.error(f"未知模型：{model_name}")
        return None, None, device, None

    model_config = AVAILABLE_MODELS[model_name]
    model_path = model_config["path"]

    try:
        if model_name == "GLM-4V-9B":
            # GLM-4V模型加载
            model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            )
            processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            return model, processor, device, "glm4v"

        elif model_name == "MedGemma-4B":
            # MedGemma模型使用transformers加载
            from transformers import AutoModelForCausalLM
            from transformers import AutoProcessor as MedGemmaProcessor

            # 检查本地路径是否存在
            if not os.path.exists(model_path):
                # 如果本地不存在，尝试在线加载
                model_path = "google/medgemma-4b-it"

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
            processor = MedGemmaProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            return model, processor, device, "medgemma"

    except Exception as e:
        st.error(f"模型加载失败：{e}")
        return None, None, device, None


def get_device_info():
    """获取核心设备信息"""
    import psutil
    
    device_info = []
    
    # GPU信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        device_info.append({"项目": "🎮 GPU", "信息": f"{gpu_name} ({gpu_memory})"})
        device_info.append({"项目": "🔧 CUDA", "信息": f"✅ {torch.version.cuda}" if torch.version.cuda else "✅ 可用"})
    else:
        device_info.append({"项目": "🎮 GPU", "信息": "❌ 不可用 (使用CPU)"})
    
    # CPU信息
    cpu_count = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    device_info.append({"项目": "💻 CPU", "信息": f"{cpu_count} 核心 ({cpu_usage:.1f}% 使用中)"})
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024**3)
    memory_usage = memory.percent
    device_info.append({"项目": "🧠 内存", "信息": f"{memory_total:.1f} GB ({memory_usage:.1f}% 使用中)"})
    
    return device_info


def validate_image(uploaded_file):
    """验证上传的图像文件"""
    if uploaded_file is None:
        return False, "未选择文件"

    # 检查文件扩展名
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        return (
            False,
            f"不支持的格式 {file_ext}。支持的格式：{', '.join(SUPPORTED_FORMATS)}",
        )

    # 检查文件大小（限制为10MB）
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "文件大小超过10MB限制"

    return True, "文件验证通过"


def parse_analysis_result(raw_text):
    """解析分析结果，分离think和answer部分"""

    # 查找think和answer标签
    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_text, re.DOTALL)

    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else ""

    # 如果没有找到标签，返回原始文本作为answer
    if not think_content and not answer_content:
        answer_content = raw_text.strip()

    return think_content, answer_content


def process_image_analysis(
    image,
    model,
    processor,
    device,
    model_type="glm4v",
    output_container=None,
    image_name="unknown.jpg",
    model_name="Unknown",
    enable_cache=True,
):
    """处理图像分析（统一流式输出，支持缓存）"""
    prompt = DEFAULT_MEDICAL_PROMPT

    # 检查缓存
    if enable_cache:
        cached_result = get_cached_result(image, model_type, model_name, prompt)
        if cached_result:
            if output_container:
                with output_container:
                    st.success("🎯 **从缓存中获取结果（避免重复计算）**")
                    st.info(f"📅 缓存时间：{cached_result['created_at']}")
                    st.markdown(
                        f"⏱️ **原处理时间：** {cached_result['processing_time']:.2f} 秒"
                    )

            return cached_result["raw_result"]

    try:
        if model_type == "glm4v":
            # GLM-4V模型处理方式
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

            # 准备输入
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # 流式生成
            streamer = TextIteratorStreamer(
                processor.tokenizer, skip_prompt=True, skip_special_tokens=False
            )

            generation_kwargs = dict(
                inputs,
                max_new_tokens=8192,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                streamer=streamer,
            )

            # 在单独的线程中运行生成
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # 流式显示生成的文本
            generated_text = ""
            if output_container:
                with output_container:
                    st.write("🤖 **AI正在生成诊断结果...**")
                    text_placeholder = st.empty()

                    for new_text in streamer:
                        generated_text += new_text
                        # 实时更新显示
                        text_placeholder.write(generated_text)

            else:
                # 如果没有容器，只收集文本
                for new_text in streamer:
                    generated_text += new_text

            thread.join()
            return generated_text.strip()

        elif model_type == "medgemma":
            # MedGemma模型使用transformers方式处理
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

            # 准备输入
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # 流式生成
            streamer = TextIteratorStreamer(
                processor.tokenizer, skip_prompt=True, skip_special_tokens=False
            )

            generation_kwargs = dict(
                inputs,
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                streamer=streamer,
            )

            # 在单独的线程中运行生成
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # 流式显示生成的文本
            generated_text = ""
            if output_container:
                with output_container:
                    st.write("🤖 **MedGemma正在生成专业诊断报告...**")
                    text_placeholder = st.empty()

                    for new_text in streamer:
                        generated_text += new_text
                        # 实时更新显示
                        text_placeholder.write(generated_text)
            else:
                # 如果没有容器，只收集文本
                for new_text in streamer:
                    generated_text += new_text

            thread.join()

            # 保存到缓存
            if enable_cache and generated_text.strip():
                think_content, answer_content = parse_analysis_result(generated_text)
                # 这里需要传入处理时间，暂时设为0，在调用处会更新
                save_analysis_result(
                    image,
                    image_name,
                    model_type,
                    model_name,
                    prompt,
                    generated_text,
                    think_content,
                    answer_content,
                    0,
                )

            return generated_text.strip()

    except Exception as e:
        if output_container:
            with output_container:
                st.error(f"❌ 分析过程中出现错误：{str(e)}")
        return f"分析过程中出现错误：{str(e)}"


# 主标题
st.markdown(
    '<h1 class="main-header">🏥 医学图像AI诊断系统</h1>', unsafe_allow_html=True
)

# 侧边栏 - 系统信息和配置
with st.sidebar:
    st.header("🔧 系统配置")

    # 设备信息
    st.subheader("💻 设备信息")
    device_info = get_device_info()
    
    # 显示核心设备信息表格
    st.table(device_info)

    st.divider()

    # 模型配置
    st.subheader("🤖 模型配置")

    # 模型选择
    selected_model = st.selectbox(
        "选择AI模型",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.current_model),
        format_func=lambda x: f"{x} - {AVAILABLE_MODELS[x]['description']}",
        help="选择用于图像分析的AI模型",
    )

    # 显示选中模型的详细信息
    model_info = AVAILABLE_MODELS[selected_model]
    st.info(f"**模型类型：** {model_info['type']}\n**路径：** {model_info['path']}")

    # 检查MedGemma模型是否存在
    if selected_model == "MedGemma-4B":
        medgemma_path = model_info["path"]
        if not os.path.exists(medgemma_path):
            st.warning(f"⚠️ MedGemma模型路径不存在：{medgemma_path}")
            st.info("💡 请确保已下载MedGemma模型到指定路径")

    # 模型加载状态
    if (
        not st.session_state.model_loaded
        or st.session_state.current_model != selected_model
    ):
        if st.button(f"🚀 加载 {selected_model} 模型", type="primary"):
            with st.spinner(f"正在加载 {selected_model} 模型，请稍候..."):
                model, processor, device, model_type = load_model(selected_model)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.processor = processor
                    st.session_state.device = device
                    st.session_state.current_model = selected_model
                    st.session_state.model_type = model_type
                    st.session_state.model_loaded = True
                    st.success(f"✅ {selected_model} 模型加载成功！")
                    st.rerun()
    else:
        st.success(f"✅ {st.session_state.current_model} 模型已就绪")
        if st.button("🔄 重新加载模型"):
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.processor = None
            st.session_state.device = None
            st.session_state.model_type = "glm4v"
            st.rerun()

    st.divider()

# 主界面
if not st.session_state.model_loaded:
    st.warning("⚠️ 请先在侧边栏加载GLM-4V模型")

else:
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🖼️ 单张图像分析", "📊 批量处理", "📋 历史结果", "🗃️ 缓存管理"]
    )

    with tab1:
        st.header("单张图像分析")

        # 文件上传
        uploaded_file = st.file_uploader(
            "选择医学图像文件",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            help="支持常见的图像格式，文件大小限制10MB",
        )

        if uploaded_file is not None:
            # 验证文件
            is_valid, message = validate_image(uploaded_file)

            if is_valid:
                st.success(f"✅ {message}")

                # 创建两列布局
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📷 原始图像")

                    # 显示图像
                    image = Image.open(uploaded_file)
                    st.image(
                        image, caption=f"文件名: {uploaded_file.name}", width="stretch"
                    )

                    # 图像信息
                    st.markdown(
                        f"""
                    <div class="image-info">
                    📏 <strong>尺寸：</strong> {image.size[0]} × {image.size[1]} 像素<br>
                    🎨 <strong>模式：</strong> {image.mode}<br>
                    📁 <strong>大小：</strong> {uploaded_file.size / 1024:.1f} KB<br>
                    📝 <strong>格式：</strong> {image.format}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.subheader("🤖 AI诊断分析")

                    if st.button("🔍 开始分析", type="primary"):
                        # 创建流式输出容器
                        streaming_container = st.empty()
                        result_containers = st.container()

                        try:
                            # 处理图像
                            start_time = time.time()

                            # 调用流式分析
                            generated_text = process_image_analysis(
                                image,
                                st.session_state.model,
                                st.session_state.processor,
                                st.session_state.device,
                                st.session_state.model_type,
                                streaming_container,
                                uploaded_file.name,
                                st.session_state.current_model,
                                True,  # enable_cache
                            )

                            end_time = time.time()
                            processing_time = end_time - start_time

                            # 更新缓存中的处理时间
                            update_processing_time(
                                image,
                                st.session_state.model_type,
                                st.session_state.current_model,
                                DEFAULT_MEDICAL_PROMPT,
                                processing_time,
                            )

                            # 清空流式输出容器
                            streaming_container.empty()

                            # 解析结果
                            think_content, answer_content = parse_analysis_result(
                                generated_text
                            )

                            # 显示最终结果
                            with result_containers:
                                st.markdown(
                                    f"**⏱️ 处理时间：** {processing_time:.2f} 秒"
                                )

                                if answer_content:
                                    with st.expander("📋 诊断结果", expanded=True):
                                        st.write(answer_content)

                                if think_content:
                                    with st.expander("🤔 分析过程", expanded=False):
                                        st.write(think_content)

                            # 保存到历史记录
                            result_data = {
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "filename": uploaded_file.name,
                                "image_size": f"{image.size[0]}x{image.size[1]}",
                                "processing_time": f"{processing_time:.2f}s",
                                "result": generated_text,
                                "think_content": think_content,
                                "answer_content": answer_content,
                            }
                            st.session_state.analysis_results.append(result_data)

                            # 提供下载选项
                            download_content = "医学图像诊断报告\n"
                            download_content += "=" * 50 + "\n"
                            download_content += f"文件名：{uploaded_file.name}\n"
                            download_content += f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            download_content += f"处理时间：{processing_time:.2f}秒\n"
                            download_content += (
                                f"图像尺寸：{image.size[0]}x{image.size[1]}像素\n\n"
                            )

                            if answer_content:
                                download_content += (
                                    f"诊断结果：\n{'-' * 30}\n{answer_content}\n\n"
                                )

                            if think_content:
                                download_content += (
                                    f"分析过程：\n{'-' * 30}\n{think_content}\n"
                                )

                            st.download_button(
                                label="📥 下载诊断报告",
                                data=download_content,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_诊断报告.txt",
                                mime="text/plain",
                            )

                        except Exception as e:
                            streaming_container.empty()
                            st.error(f"❌ 分析失败：{str(e)}")
            else:
                st.error(f"❌ {message}")

    with tab2:
        st.header("批量图像处理")

        # 批量文件上传
        uploaded_files = st.file_uploader(
            "选择多张医学图像文件",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            accept_multiple_files=True,
            help="可以同时选择多张图像进行批量分析",
        )

        if uploaded_files:
            st.success(f"✅ 已选择 {len(uploaded_files)} 个文件")

            # 显示文件列表
            with st.expander("📁 查看选择的文件", expanded=True):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")

            start_batch = st.button("🚀 开始批量分析", type="primary")

            if start_batch:
                # 批量处理
                total_start_time = time.time()

                # 创建进度显示
                overall_progress = st.progress(0)
                status_container = st.empty()
                results_container = st.container()

                batch_results = []

                for i, uploaded_file in enumerate(uploaded_files):
                    # 更新整体进度
                    progress = (i) / len(uploaded_files)
                    overall_progress.progress(progress)
                    status_container.write(
                        f"🔄 正在处理第 {i + 1}/{len(uploaded_files)} 个文件: {uploaded_file.name}"
                    )

                    # 验证文件
                    is_valid, message = validate_image(uploaded_file)

                    if is_valid:
                        try:
                            # 加载图像
                            image = Image.open(uploaded_file)

                            # 处理图像
                            start_time = time.time()
                            generated_text = process_image_analysis(
                                image,
                                st.session_state.model,
                                st.session_state.processor,
                                st.session_state.device,
                                st.session_state.model_type,
                                None,  # output_container
                                uploaded_file.name,
                                st.session_state.current_model,
                                True,  # enable_cache
                            )
                            end_time = time.time()
                            processing_time = end_time - start_time

                            # 更新缓存中的处理时间
                            update_processing_time(
                                image,
                                st.session_state.model_type,
                                st.session_state.current_model,
                                DEFAULT_MEDICAL_PROMPT,
                                processing_time,
                            )

                            # 解析结果
                            think_content, answer_content = parse_analysis_result(
                                generated_text
                            )

                            # 保存结果
                            result_data = {
                                "filename": uploaded_file.name,
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "image_size": f"{image.size[0]}x{image.size[1]}",
                                "processing_time": f"{processing_time:.2f}s",
                                "result": generated_text,
                                "think_content": think_content,
                                "answer_content": answer_content,
                                "status": "success",
                            }
                            batch_results.append(result_data)
                            st.session_state.analysis_results.append(result_data)

                            # 显示单个结果
                            with results_container:
                                with st.expander(
                                    f"✅ {uploaded_file.name} - 分析完成 ({processing_time:.2f}s)"
                                ):
                                    col_img, col_result = st.columns([1, 2])
                                    with col_img:
                                        st.image(
                                            image,
                                            caption=uploaded_file.name,
                                            width="stretch",
                                        )
                                    with col_result:
                                        if answer_content:
                                            with st.expander(
                                                "📋 诊断结果", expanded=True
                                            ):
                                                st.write(answer_content)
                                        if think_content:
                                            with st.expander(
                                                "🤔 分析过程", expanded=False
                                            ):
                                                st.write(think_content)

                        except Exception as e:
                            result_data = {
                                "filename": uploaded_file.name,
                                "timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "error": str(e),
                                "status": "error",
                            }
                            batch_results.append(result_data)

                            with results_container:
                                st.error(f"❌ {uploaded_file.name} 处理失败: {str(e)}")
                    else:
                        result_data = {
                            "filename": uploaded_file.name,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "error": message,
                            "status": "invalid",
                        }
                        batch_results.append(result_data)

                        with results_container:
                            st.warning(f"⚠️ {uploaded_file.name}: {message}")

                # 完成批量处理
                overall_progress.progress(1.0)
                total_end_time = time.time()
                total_time = total_end_time - total_start_time

                status_container.success(
                    f"🎉 批量处理完成！总耗时：{total_time:.2f}秒，平均每张：{total_time / len(uploaded_files):.2f}秒"
                )

                # 提供批量下载
                if batch_results:
                    # 创建批量报告
                    batch_report = f"批量医学图像分析报告\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    batch_report += f"总文件数：{len(uploaded_files)}\n"
                    batch_report += f"总处理时间：{total_time:.2f}秒\n"
                    batch_report += (
                        f"平均处理时间：{total_time / len(uploaded_files):.2f}秒\n"
                    )
                    batch_report += "=" * 80 + "\n\n"

                    for result in batch_results:
                        batch_report += f"文件名：{result['filename']}\n"
                        batch_report += f"时间戳：{result['timestamp']}\n"
                        if result["status"] == "success":
                            batch_report += f"图像尺寸：{result['image_size']}\n"
                            batch_report += f"处理时间：{result['processing_time']}\n"
                            batch_report += f"诊断结果：\n{result['result']}\n"
                        else:
                            batch_report += "状态：处理失败\n"
                            batch_report += f"错误：{result.get('error', '未知错误')}\n"
                        batch_report += "\n" + "-" * 80 + "\n\n"

                    st.download_button(
                        label="📦 下载批量分析报告",
                        data=batch_report,
                        file_name=f"批量分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )

    with tab3:
        st.header("历史分析结果")

        if st.session_state.analysis_results:
            # 统计信息
            total_analyses = len(st.session_state.analysis_results)
            successful_analyses = sum(
                1
                for r in st.session_state.analysis_results
                if r.get("status") != "error"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总分析次数", total_analyses)
            with col2:
                st.metric("成功分析", successful_analyses)
            with col3:
                st.metric(
                    "成功率", f"{successful_analyses / total_analyses * 100:.1f}%"
                )

            st.divider()

            # 清空历史记录按钮
            if st.button("🗑️ 清空历史记录"):
                st.session_state.analysis_results = []
                st.rerun()

            # 显示历史结果
            for i, result in enumerate(reversed(st.session_state.analysis_results)):
                with st.expander(
                    f"📄 {result['filename']} - {result['timestamp']}", expanded=False
                ):
                    if result.get("status") == "success":
                        col_info, col_result = st.columns([1, 2])

                        with col_info:
                            st.write(f"**文件名：** {result['filename']}")
                            st.write(f"**时间：** {result['timestamp']}")
                            if "image_size" in result:
                                st.write(f"**尺寸：** {result['image_size']}")
                            if "processing_time" in result:
                                st.write(f"**处理时间：** {result['processing_time']}")

                        with col_result:
                            # 显示分离的结果
                            if result.get("answer_content"):
                                with st.expander("📋 诊断结果", expanded=True):
                                    st.write(result["answer_content"])

                            if result.get("think_content"):
                                with st.expander("🤔 分析过程", expanded=False):
                                    st.write(result["think_content"])

                            # 如果没有分离的内容，显示原始结果
                            if not result.get("answer_content") and not result.get(
                                "think_content"
                            ):
                                st.write("**诊断结果：**")
                                st.write(result["result"])

                            # 单个结果下载
                            download_data = "医学图像诊断报告\n"
                            download_data += "=" * 50 + "\n"
                            download_data += f"文件名：{result['filename']}\n"
                            download_data += f"分析时间：{result['timestamp']}\n"
                            if "processing_time" in result:
                                download_data += (
                                    f"处理时间：{result['processing_time']}\n"
                                )
                            download_data += "\n"

                            if result.get("answer_content"):
                                download_data += f"诊断结果：\n{'-' * 30}\n{result['answer_content']}\n\n"

                            if result.get("think_content"):
                                download_data += f"分析过程：\n{'-' * 30}\n{result['think_content']}\n"

                            if not result.get("answer_content") and not result.get(
                                "think_content"
                            ):
                                download_data += f"诊断结果：\n{result['result']}\n"

                            st.download_button(
                                label="📥 下载此报告",
                                data=download_data,
                                file_name=f"{os.path.splitext(result['filename'])[0]}_诊断报告.txt",
                                mime="text/plain",
                                key=f"download_{i}",
                            )
                    else:
                        st.error(f"处理失败：{result.get('error', '未知错误')}")
        else:
            st.info("📭 暂无历史分析结果")

    with tab4:
        st.header("缓存管理")

        st.markdown("""
        ### 🗃️ 数据库缓存系统
        
        系统使用SQLite数据库缓存分析结果，相同图片和提示词组合将直接返回已有结果，避免重复计算。
        
        **缓存机制：**
        - 🔍 **图片识别**：基于图片内容的MD5哈希值
        - 📝 **提示词匹配**：基于提示词内容的MD5哈希值  
        - 🤖 **模型区分**：不同模型的结果分别缓存
        - ⚡ **快速查询**：数据库索引优化查询性能
        """)

        # 获取缓存统计信息
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            total_records = cursor.fetchone()[0]

            # 按模型分组统计
            cursor.execute("""
                SELECT model_name, COUNT(*) 
                FROM analysis_results 
                GROUP BY model_name
            """)
            model_stats = cursor.fetchall()

            # 最近7天的记录数
            cursor.execute("""
                SELECT COUNT(*) FROM analysis_results 
                WHERE created_at >= datetime('now', '-7 days')
            """)
            recent_records = cursor.fetchone()[0]

            # 数据库大小
            cursor.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            db_size = cursor.fetchone()[0] / 1024 / 1024  # MB

            conn.close()

            # 显示统计信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总缓存记录", total_records)
            with col2:
                st.metric("近7天记录", recent_records)
            with col3:
                st.metric("数据库大小", f"{db_size:.2f} MB")
            with col4:
                st.metric("缓存文件", DATABASE_PATH)

            # 模型使用统计
            if model_stats:
                st.subheader("📊 模型使用统计")
                for model_name, count in model_stats:
                    st.write(f"**{model_name}**: {count} 次分析")

            st.divider()

            # 缓存历史记录
            st.subheader("📋 缓存历史记录")

            # 获取记录数量选择
            record_limit = st.selectbox("显示记录数", [10, 25, 50, 100], index=1)

            history_records = get_analysis_history(record_limit)

            if history_records:
                for i, record in enumerate(history_records):
                    with st.expander(
                        f"📄 {record['image_name']} - {record['created_at'][:19]}",
                        expanded=False,
                    ):
                        col_info, col_result = st.columns([1, 2])

                        with col_info:
                            st.write(f"**文件名：** {record['image_name']}")
                            st.write(f"**图片尺寸：** {record['image_size']}")
                            st.write(
                                f"**使用模型：** {record['model_name']} ({record['model_type']})"
                            )
                            st.write(
                                f"**处理时间：** {record['processing_time']:.2f}秒"
                            )
                            st.write(f"**缓存时间：** {record['created_at'][:19]}")

                        with col_result:
                            # 显示分离的结果
                            if record.get("answer_content"):
                                with st.expander("📋 诊断结果", expanded=True):
                                    st.write(record["answer_content"])

                            if record.get("think_content"):
                                with st.expander("🤔 分析过程", expanded=False):
                                    st.write(record["think_content"])
            else:
                st.info("📭 暂无缓存记录")

            st.divider()

            # 缓存管理操作
            st.subheader("🛠️ 缓存管理操作")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🗑️ 清空所有缓存", type="secondary"):
                    if clear_analysis_cache():
                        st.success("✅ 缓存已清空")
                        st.rerun()
                    else:
                        st.error("❌ 清空缓存失败")

            with col2:
                # 导出缓存数据
                if st.button("📤 导出缓存数据"):
                    try:
                        import pandas as pd

                        conn = sqlite3.connect(DATABASE_PATH)
                        df = pd.read_sql_query(
                            "SELECT * FROM analysis_results ORDER BY created_at DESC",
                            conn,
                        )
                        conn.close()

                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="📥 下载CSV文件",
                            data=csv_data,
                            file_name=f"analysis_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                        st.success("✅ 导出数据准备完成")
                    except ImportError:
                        st.warning("⚠️ 需要安装pandas库才能导出CSV")
                    except Exception as e:
                        st.error(f"❌ 导出失败: {e}")

        except Exception as e:
            st.error(f"❌ 获取缓存信息失败：{e}")


# 更新todo状态
if st.session_state.model_loaded:
    # 标记设计和创建任务完成
    pass
