#!/usr/bin/env python3
"""
医学图像AI诊断系统使用演示
展示如何在代码中使用核心功能
"""

import os
import sys
from PIL import Image
import torch
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_image_analysis():
    """演示图像分析功能"""
    print("🏥 医学图像AI诊断系统 - 功能演示")
    print("=" * 60)
    
    # 检查sample目录
    sample_dir = "sample"
    if not os.path.exists(sample_dir):
        print(f"❌ 示例目录 '{sample_dir}' 不存在")
        return
    
    # 获取示例图像
    image_files = []
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for filename in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                image_files.append(file_path)
    
    if not image_files:
        print(f"❌ 在 '{sample_dir}' 中未找到支持的图像文件")
        return
    
    print(f"🔍 发现 {len(image_files)} 个示例图像文件：")
    for i, file_path in enumerate(image_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    print("\n📱 Web界面功能演示：")
    print("=" * 40)
    
    print("1. 🖼️ 单张图像分析")
    print("   • 上传医学图像文件")
    print("   • 实时预览图像信息")
    print("   • AI智能诊断分析")
    print("   • 流式结果显示")
    print("   • 下载诊断报告")
    
    print("\n2. 📊 批量处理")
    print("   • 多文件同时上传")
    print("   • 批量AI诊断")
    print("   • 进度实时跟踪")
    print("   • 综合报告生成")
    
    print("\n3. 📋 历史结果")
    print("   • 查看分析历史")
    print("   • 统计成功率")
    print("   • 重新下载报告")
    print("   • 清理历史记录")
    
    print("\n4. ⚙️ 系统配置")
    print("   • GPU/CPU设备信息")
    print("   • 模型加载管理")
    print("   • 自定义提示词")
    print("   • 参数调节控制")

def demo_streamlit_features():
    """演示Streamlit UI特性"""
    print("\n🎨 Streamlit UI特性：")
    print("=" * 40)
    
    features = [
        ("📱 响应式设计", "适配不同屏幕尺寸，支持移动端访问"),
        ("🎯 直观操作", "拖拽上传文件，一键分析图像"),
        ("⚡ 实时反馈", "流式显示分析过程，即时状态更新"),
        ("🎨 现代界面", "美观的UI设计，清晰的信息展示"),
        ("🔄 智能缓存", "模型缓存机制，避免重复加载"),
        ("📊 数据可视化", "图像信息展示，统计图表显示"),
        ("💾 结果管理", "自动保存分析结果，支持批量下载"),
        ("🛡️ 错误处理", "友好的错误提示，异常情况处理"),
    ]
    
    for title, description in features:
        print(f"  {title}")
        print(f"    {description}")
    
    print(f"\n🚀 启动命令：")
    print(f"   python run_app.py")
    print(f"   或者: streamlit run streamlit_app.py")
    
    print(f"\n🔗 访问地址：")
    print(f"   http://localhost:8501")

def show_file_structure():
    """显示项目文件结构"""
    print("\n📁 项目文件结构：")
    print("=" * 40)
    
    structure = [
        ("streamlit_app.py", "主要的Streamlit Web应用"),
        ("vision.py", "原始的命令行图像分析脚本"),
        ("run_app.py", "应用启动脚本（跨平台）"),
        ("start_app.bat", "Windows批处理启动脚本"),
        ("requirements.txt", "Python依赖包列表"),
        ("README_streamlit.md", "详细的使用说明文档"),
        ("demo_usage.py", "功能演示脚本（当前文件）"),
        ("check_cuda.py", "CUDA环境检查脚本"),
        ("sample/", "示例医学图像目录"),
    ]
    
    for filename, description in structure:
        print(f"  📄 {filename:<20} - {description}")

def main():
    """主函数"""
    demo_image_analysis()
    demo_streamlit_features()
    show_file_structure()
    
    print(f"\n🎯 下一步操作：")
    print(f"=" * 40)
    print(f"1. 确保在 transformers conda 环境中")
    print(f"2. 安装依赖：pip install -r requirements.txt")
    print(f"3. 启动应用：python run_app.py")
    print(f"4. 在浏览器中打开：http://localhost:8501")
    print(f"5. 在侧边栏加载GLM-4V模型")
    print(f"6. 上传医学图像开始分析")
    
    print(f"\n💡 提示：首次运行会下载约18GB的模型文件，请确保网络畅通。")

if __name__ == "__main__":
    main()
