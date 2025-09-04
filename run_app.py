#!/usr/bin/env python3
"""
医学图像AI诊断系统启动脚本
运行此脚本来启动Streamlit Web应用
"""

import subprocess
import sys
import os

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"✅ Python版本：{python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = [
        'streamlit', 'torch', 'transformers', 'modelscope', 'PIL'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下包：{', '.join(missing_packages)}")
        print("请先安装缺少的包，然后重新运行此脚本。")
        return False
    
    return True

def main():
    """主函数"""
    print("🏥 医学图像AI诊断系统")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 检查streamlit_app.py是否存在
    if not os.path.exists("streamlit_app.py"):
        print("❌ 错误：找不到 streamlit_app.py 文件")
        sys.exit(1)
    
    print("\n🚀 启动Streamlit应用...")
    print("📱 应用将在浏览器中自动打开")
    print("🔗 如果没有自动打开，请访问：http://localhost:8501")
    print("\n💡 提示：")
    print("   - 使用 Ctrl+C 停止应用")
    print("   - 确保在transformers conda环境中运行")
    print("   - 确保GPU可用以获得最佳性能")
    print("\n" + "=" * 50)
    
    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
