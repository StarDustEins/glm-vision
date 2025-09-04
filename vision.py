from modelscope import AutoProcessor, Glm4vForConditionalGeneration
import torch
import time
import os
from PIL import Image
import sys
from transformers import TextIteratorStreamer

# 设置使用GPU (RTX 5090)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# 设置图片目录路径
SAMPLE_DIR = "sample"

# 支持的图片格式
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_image_files(directory):
    """获取目录下所有支持的图片文件"""
    if not os.path.exists(directory):
        print(f"❌ 错误：目录 '{directory}' 不存在！")
        return []

    image_files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_FORMATS:
                image_files.append(file_path)

    return sorted(image_files)  # 按文件名排序


def load_local_image(image_path):
    """加载本地图片文件"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：图片文件 '{image_path}' 不存在！")
        return None

    # 检查文件扩展名
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        print(f"❌ 错误：不支持的图片格式 '{file_ext}'")
        print(f"💡 支持的格式：{', '.join(SUPPORTED_FORMATS)}")
        return None

    # 加载图片
    try:
        image = Image.open(image_path)
        print(f"✅ 成功加载图片：{os.path.basename(image_path)}")
        print(f"📏 图片尺寸：{image.size[0]} x {image.size[1]} 像素")
        print(f"🎨 图片模式：{image.mode}")
        print(f"📁 文件大小：{os.path.getsize(image_path) / 1024:.1f} KB")
        return image
    except Exception as e:
        print(f"❌ 加载图片失败：{e}")
        return None


# 获取所有图片文件
image_files = get_image_files(SAMPLE_DIR)
if not image_files:
    print(f"❌ 在目录 '{SAMPLE_DIR}' 中没有找到支持的图片文件！")
    print(f"💡 支持的格式：{', '.join(SUPPORTED_FORMATS)}")
    exit(1)

print(f"🔍 发现 {len(image_files)} 个图片文件：")
for i, file_path in enumerate(image_files, 1):
    print(f"  {i}. {os.path.basename(file_path)}")
print()

MODEL_PATH = "ZhipuAI/GLM-4.1V-9B-Thinking"

# 加载模型和处理器（只加载一次）
print("🤖 正在加载模型...")
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # 显式指定使用第一个GPU (RTX 5090)
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
print("✅ 模型加载完成！\n")


def stream_generate(model, processor, inputs, max_new_tokens=8192):
    """流式生成文本"""
    print("\n" + "=" * 60)
    print("🤖 AI 图片描述结果（流式输出）：")
    print("=" * 60)

    # 获取输入长度
    input_length = inputs["input_ids"].shape[1]

    # 使用模型的generate方法进行流式生成
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=False
    )

    # 生成参数
    generation_kwargs = dict(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        streamer=streamer,
    )

    # 在单独的线程中运行生成
    import threading

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 实时输出生成的文本
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end="", flush=True)

    thread.join()

    print("\n" + "=" * 60)
    return generated_text


def process_single_image(image_path, model, processor, device):
    """处理单个图片"""
    print(f"\n{'=' * 80}")
    print(f"🖼️  正在处理：{os.path.basename(image_path)}")
    print(f"{'=' * 80}")

    # 加载图片
    image = load_local_image(image_path)
    if image is None:
        print(f"⚠️  跳过文件：{os.path.basename(image_path)}")
        return

    # 准备消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # 使用PIL Image对象
                },
                {
                    "type": "text",
                    "text": "你是医学阅片专家，仔细查看检查图片，给出详细的检查诊断结果",
                },
            ],
        }
    ]

    # 准备输入
    time_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)  # 确保输入数据也在GPU上

    # 流式生成
    print("🚀 开始生成...")
    generated_text = stream_generate(model, processor, inputs, max_new_tokens=8192)

    time_end = time.time()
    print(f"⏱️  处理时间：{time_end - time_start:.2f} 秒")

    # 保存结果到文件（可选）
    output_file = os.path.splitext(image_path)[0] + "_description.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"图片：{os.path.basename(image_path)}\n")
            f.write(f"处理时间：{time_end - time_start:.2f} 秒\n")
            f.write(f"描述：\n{generated_text.strip()}\n")
        print(f"💾 描述已保存到：{os.path.basename(output_file)}")
    except Exception as e:
        print(f"⚠️  保存文件失败：{e}")


# 批量处理所有图片
total_start_time = time.time()
print(f"🎯 开始批量处理 {len(image_files)} 个图片文件...")

for i, image_path in enumerate(image_files, 1):
    print(f"\n{'🔄' * 20} 进度：{i}/{len(image_files)} {'🔄' * 20}")
    process_single_image(image_path, model, processor, device)

total_end_time = time.time()
print(f"\n{'=' * 80}")
print(f"🎉 批量处理完成！")
print(f"📊 总共处理了 {len(image_files)} 个图片")
print(f"⏱️  总耗时：{total_end_time - total_start_time:.2f} 秒")
print(
    f"⚡ 平均每张图片：{(total_end_time - total_start_time) / len(image_files):.2f} 秒"
)
print(f"{'=' * 80}")
