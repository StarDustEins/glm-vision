#!/usr/bin/env python3
"""
医学图像AI诊断系统测试脚本
验证系统各组件是否正常工作
"""

import os
import sys
import torch
from PIL import Image
import importlib.util

def test_environment():
    """测试运行环境"""
    print("🔍 测试运行环境...")
    print("-" * 40)
    
    # Python版本
    python_version = sys.version_info
    print(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 必要包测试
    packages = {
        'streamlit': 'Streamlit Web框架',
        'torch': 'PyTorch深度学习框架',
        'transformers': 'Hugging Face Transformers',
        'modelscope': 'ModelScope模型库',
        'PIL': 'Pillow图像处理库'
    }
    
    missing_packages = []
    for package, description in packages.items():
        try:
            if package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: 未安装 - {description}")
    
    if missing_packages:
        print(f"\n⚠️ 缺少包：{', '.join(missing_packages)}")
        print("请运行：pip install -r requirements.txt")
        return False
    
    return True

def test_cuda():
    """测试CUDA环境"""
    print(f"\n🚀 测试CUDA环境...")
    print("-" * 40)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"📱 GPU设备数量: {torch.cuda.device_count()}")
        print(f"🎯 当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print(f"⚠️ CUDA不可用，将使用CPU模式")
        print(f"💡 安装CUDA以获得更好性能")
        return False

def test_sample_images():
    """测试示例图像"""
    print(f"\n📷 测试示例图像...")
    print("-" * 40)
    
    sample_dir = "sample"
    if not os.path.exists(sample_dir):
        print(f"❌ 示例目录 '{sample_dir}' 不存在")
        return False
    
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = []
    
    for filename in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                try:
                    image = Image.open(file_path)
                    image_files.append({
                        'path': file_path,
                        'name': filename,
                        'size': image.size,
                        'mode': image.mode,
                        'format': image.format
                    })
                    print(f"✅ {filename}: {image.size[0]}x{image.size[1]}, {image.mode}")
                except Exception as e:
                    print(f"❌ {filename}: 加载失败 - {e}")
    
    if image_files:
        print(f"\n📊 找到 {len(image_files)} 个有效图像文件")
        return True
    else:
        print(f"❌ 未找到有效的图像文件")
        return False

def test_streamlit_app():
    """测试Streamlit应用文件"""
    print(f"\n📱 测试Streamlit应用...")
    print("-" * 40)
    
    app_file = "streamlit_app.py"
    if not os.path.exists(app_file):
        print(f"❌ 应用文件 '{app_file}' 不存在")
        return False
    
    # 检查文件语法
    try:
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单的语法检查
        compile(content, app_file, 'exec')
        print(f"✅ {app_file}: 语法检查通过")
        
        # 检查关键组件
        key_components = [
            'st.set_page_config',
            'st.file_uploader',
            'st.image',
            'st.tabs',
            'process_image_analysis',
            'load_model'
        ]
        
        missing_components = []
        for component in key_components:
            if component in content:
                print(f"✅ 包含组件: {component}")
            else:
                missing_components.append(component)
                print(f"❌ 缺少组件: {component}")
        
        if not missing_components:
            print(f"✅ 所有关键组件都已包含")
            return True
        else:
            print(f"⚠️ 缺少 {len(missing_components)} 个组件")
            return False
            
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def show_usage_instructions():
    """显示使用说明"""
    print(f"\n📖 使用说明：")
    print("=" * 60)
    
    print("1. 🚀 启动系统")
    print("   conda activate transformers")
    print("   python run_app.py")
    print()
    
    print("2. 🌐 访问Web界面")
    print("   浏览器打开: http://localhost:8501")
    print()
    
    print("3. 🤖 加载模型")
    print("   在侧边栏点击'加载GLM-4V模型'")
    print("   等待模型下载和加载完成")
    print()
    
    print("4. 🖼️ 分析图像")
    print("   单张分析：上传图像 → 点击分析 → 查看结果")
    print("   批量分析：选择多个文件 → 批量处理 → 下载报告")
    print()
    
    print("5. 📋 管理结果")
    print("   历史记录：查看所有分析历史")
    print("   下载报告：保存诊断结果到本地")

def main():
    """主测试函数"""
    print("🏥 医学图像AI诊断系统 - 系统测试")
    print("=" * 60)
    
    # 运行各项测试
    env_ok = test_environment()
    cuda_ok = test_cuda()
    images_ok = test_sample_images()
    app_ok = test_streamlit_app()
    
    print(f"\n📊 测试结果汇总：")
    print("=" * 60)
    print(f"{'环境依赖':<15}: {'✅ 通过' if env_ok else '❌ 失败'}")
    print(f"{'CUDA支持':<15}: {'✅ 可用' if cuda_ok else '⚠️ 不可用'}")
    print(f"{'示例图像':<15}: {'✅ 正常' if images_ok else '❌ 异常'}")
    print(f"{'应用文件':<15}: {'✅ 正常' if app_ok else '❌ 异常'}")
    
    if all([env_ok, images_ok, app_ok]):
        print(f"\n🎉 系统测试通过！可以正常启动应用。")
        show_usage_instructions()
    else:
        print(f"\n⚠️ 系统测试发现问题，请修复后重新测试。")
        
        if not env_ok:
            print("   • 安装缺少的Python包")
        if not images_ok:
            print("   • 检查sample目录和图像文件")
        if not app_ok:
            print("   • 检查streamlit_app.py文件")
    
    print(f"\n🔗 相关文件：")
    print(f"   📄 streamlit_app.py  - 主应用文件")
    print(f"   📄 run_app.py        - 启动脚本")
    print(f"   📄 requirements.txt  - 依赖列表")
    print(f"   📖 README_streamlit.md - 详细文档")

if __name__ == "__main__":
    main()
