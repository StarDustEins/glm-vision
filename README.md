## 医学图像 AI 诊断系统（Streamlit）

基于 GLM-4V 多模态模型的医学影像分析工具，提供流式诊断报告、缓存管理与结果导出能力。本仓库已精简为单一 Streamlit 应用，可直接通过 `uv` 环境运行。

### 环境要求
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 版本 0.5.0 及以上
- GPU（推荐，需满足 Torch 对应 CUDA 条件）

### 安装依赖
首次使用时同步锁定的依赖：
```bash
uv sync
```

> 如需在隔离环境运行，可使用 `uv venv` 创建虚拟环境后执行 `uv sync`。

### 启动 Streamlit 应用
```bash
uv run streamlit run streamlit_app.py
```

常用可选参数：
- 指定地址：`--server.address 0.0.0.0`
- 指定端口：`--server.port 8501`

示例：
```bash
uv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### 使用说明
1. 首次进入应用后，在侧边栏加载所需模型（默认提供 GLM-4V-9B、MedGemma-4B）。
2. 上传医学影像文件（支持 JPG/JPEG/PNG/BMP/TIFF/WebP，≤10 MB）。
3. 查看流式生成的诊断结果，并可下载报告。
4. 在「缓存管理」页查看历史记录、导出 CSV，或清空缓存。

### 数据与缓存
- 分析结果持久化存储在 `medical_analysis_cache.db`，根据内容与提示词自动去重。
- 若需要示例图像，请自行放置到自选目录并在界面中上传。
- 删除缓存可在界面点击按钮完成，无需手动操作文件。

### 项目结构
```
├─ app_logic.py       # 业务逻辑层：模型加载、缓存、数据库操作、流式推理
├─ streamlit_app.py   # UI 层：Streamlit 页面布局、用户交互
├─ pyproject.toml     # 项目配置与依赖定义
├─ uv.lock            # 通过 uv 生成的锁文件（勿手动编辑）
├─ README.md          # 使用与开发指南
└─ .gitignore         # Git 忽略规则
```

### 架构说明
- **UI / 逻辑分离**：`streamlit_app.py` 仅负责界面渲染、用户输入与结果展示；所有模型调用、数据库缓存、统计信息由 `app_logic.py` 提供。遵循单一职责原则，便于维护与扩展。
- **缓存策略**：针对「图像内容 + 提示词 + 模型」组合生成 MD5 哈希，命中后直接读取数据库结果，避免重复推理。
- **流式推理**：使用 `TextIteratorStreamer` 逐步返回模型输出，UI 端可实时渲染诊断内容。

### 开发者指南
1. **安装依赖**：`uv sync`
2. **启动开发模式**：`uv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501`
3. **代码组织建议**：
   - 新增业务逻辑时，请在 `app_logic.py` 中实现函数，并在 `__all__` 中导出，再由 UI 层调用。
   - 涉及状态持久化的扩展（例如新增表）需同步更新 `init_database()`。
   - 如果要引入额外依赖，修改 `pyproject.toml` 后执行 `uv sync` 生成新的锁文件。
4. **模型目录约定**：UI 会优先在 `\\wsl.localhost\Ubuntu-24.04\home\elysion\.cache\modelscope\hub\models`（以及 `~/.cache/modelscope/hub/models`）下扫描本地缓存模型，再回退到远程下载。

### 常见问题 / FAQ
- **模型加载失败**：确认网络连通性、显存是否足够；MedGemma 默认尝试本地路径，若不存在会自动回退至在线模型。
- **GPU 不可用**：检查 CUDA 安装；UI 会显示当前设备信息，可在 CPU 模式下运行但性能有限。
- **缓存文件增长过快**：在 UI 的「缓存管理」标签页清空或导出数据；数据库文件默认列在 `.gitignore` 中。
- **导出 CSV 失败**：确保执行过 `uv sync`，项目依赖已包含 `pandas`。
- **自定义提示词**：可在 `DEFAULT_MEDICAL_PROMPT` 中调整，也可在 UI 层增加输入组件并传递给逻辑模块。

### 贡献与测试
- 编写新功能前请先阅读现有模块，保持风格一致（类型注解、文档字符串等）。
- 推荐引入 `pytest` 或 `streamlit testing` 方案来覆盖核心函数，可在 `tests/` 目录新增测试。
- 提交前建议运行 `uv run python -m compileall app_logic.py streamlit_app.py` 进行语法检查。
