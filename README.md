# 文物/艺术品图片识别系统

基于 Flask + WebSocket + Qwen-VL-Plus 的文物艺术品图片识别应用

## 功能特点

- 🖼️ 支持拖拽或点击上传图片
- 🔌 使用 WebSocket 实现实时通信
- 🤖 集成 Qwen-VL-Plus 多模态大模型进行识别
- 📊 置信度高于 90% 才输出识别结果
- 🎨 美观的现代化 UI 界面

## 环境要求

- Python 3.8+
- 阿里云 DashScope API Key（用于 Qwen-VL 模型）

## 安装步骤

### 1. 安装依赖

```bash
pip install flask flask-socketio dashscope requests Pillow
```

### 2. 设置 API Key

获取阿里云 DashScope API Key：https://dashscope.console.aliyun.com/

**方法一：编辑 api_key.txt 文件（推荐）**

打开 `api_key.txt` 文件，将内容修改为：
```
DASHSCOPE_API_KEY=你的 API 密钥
```

**方法二：使用环境变量**

**Windows 设置方法：**
```cmd
set DASHSCOPE_API_KEY=你的 API 密钥
```

**或使用环境变量永久设置：**
在系统环境变量中添加 `DASHSCOPE_API_KEY`

### 3. 启动应用

```bash
python app.py
```

### 4. 访问网页

打开浏览器访问：http://127.0.0.1:5000

## 使用说明

1. 点击上传区域或拖拽图片到上传框
2. 点击「开始识别」按钮
3. 等待识别结果：
   - ✅ 置信度 > 90%：显示文物名称和背景介绍
   - ❌ 置信度 ≤ 90%：显示「无法识别」

## 项目结构

```
图片识别/
├── app.py              # Flask 主应用
├── requirements.txt    # Python 依赖
├── templates/
│   └── index.html     # 前端页面
└── uploads/           # 临时上传目录（自动创建）
```

## 技术栈

- **后端**: Flask + Flask-SocketIO
- **前端**: HTML5 + CSS3 + JavaScript + Socket.IO
- **AI 模型**: Qwen-VL-Plus (阿里云 DashScope)

## 注意事项

- 图片格式支持：JPG、PNG、GIF
- 识别结果置信度阈值设为 90%
- 请确保 API Key 有效且有足够的调用额度
