@echo off
echo 🏥 医学图像AI诊断系统
echo ================================
echo.
echo 🔍 检查环境...
echo 📝 确保在transformers conda环境中运行
echo.

REM 激活conda环境
call conda activate transformers

echo 🚀 启动Streamlit应用...
echo 📱 应用将在浏览器中自动打开
echo 🔗 地址：http://localhost:8501 (或 8502)
echo.
echo 💡 使用 Ctrl+C 停止应用
echo ================================
echo.

streamlit run streamlit_app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false

pause
