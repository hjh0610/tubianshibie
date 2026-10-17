"""
文物/艺术品图片识别应用 - 基于 Flask + WebSocket + Qwen-VL
"""
import os
import base64
import json
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from dashscope import MultiModalConversation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Qwen-VL 模型配置 - 从 api_key.txt 文件或环境变量读取
def load_api_key():
    """从 api_key.txt 文件或环境变量加载 API Key"""
    # 优先从 api_key.txt 读取
    api_key_file = os.path.join(os.path.dirname(__file__), 'api_key.txt')
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('DASHSCOPE_API_KEY=') and not line.startswith('#'):
                    return line.split('=', 1)[1].strip()
    # 从环境变量读取
    return os.environ.get('DASHSCOPE_API_KEY', '')

QWEN_API_KEY = load_api_key()


def image_to_base64(image_path):
    """将图片转换为 base64 编码"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')


def recognize_artifact(image_path):
    """
    使用 Qwen-VL-Plus 模型识别文物/艺术品图片
    
    返回:
        dict: {'success': bool, 'name': str, 'background': str, 'confidence': float, 'message': str}
    """
    if not QWEN_API_KEY:
        return {
            'success': False,
            'message': '未配置 DASHSCOPE_API_KEY，请设置环境变量后重试'
        }
    
    try:
        # 将图片转换为 base64
        image_base64 = image_to_base64(image_path)
        
        # 构建请求消息
        messages = [{
            'role': 'user',
            'content': [
                {
                    'image': f'data:image/jpeg;base64,{image_base64}'
                },
                {
                    'text': '''请识别这张图片中的文物或艺术品。

【输出格式要求】
名称：[文物/艺术品的名称，如果无法识别请写"无法识别"]
背景：[历史背景、文化意义、创作年代等详细介绍，200 字以内]
确定程度：[高/中/低，三选一]

如果图片内容清晰且你能准确识别，确定程度写"高"；
如果能识别但不太确定，写"中"；
如果完全无法识别或图片不清晰，写"低"。'''
                }
            ]
        }]
        
        # 调用 Qwen-VL-Plus 模型
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages,
            api_key=QWEN_API_KEY
        )
        
        if response.status_code == 200:
            result_content = response.output.choices[0].message.content

            # 处理返回内容可能是列表的情况
            if isinstance(result_content, list):
                result_text = ' '.join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in result_content])
            elif isinstance(result_content, dict):
                result_text = result_content.get('text', str(result_content))
            else:
                result_text = str(result_content)

            # 打印调试信息
            print(f'=== 模型返回 ===\n{result_text}\n==============')

            # 解析返回结果
            name = ""
            background = ""
            confidence_level = ""
            confidence = 100  # 默认高置信度

            # 尝试从结果中提取名称、背景和确定程度
            lines = result_text.split('\n')
            for line in lines:
                line = line.strip()
                # 支持全角和半角冒号
                if '名称' in line and ('：' in line or ':' in line):
                    name = line.replace('：', ':').split(':', 1)[1].strip()
                elif '背景' in line and ('：' in line or ':' in line):
                    background = line.replace('：', ':').split(':', 1)[1].strip()
                elif '确定程度' in line and ('：' in line or ':' in line):
                    confidence_level = line.replace('：', ':').split(':', 1)[1].strip()
                    # 将确定程度转换为数值
                    if '高' in confidence_level:
                        confidence = 95
                    elif '中' in confidence_level:
                        confidence = 70
                    elif '低' in confidence_level:
                        confidence = 40

            # 打印解析结果用于调试
            print(f'解析结果：name="{name}", confidence={confidence}')

            # 检查是否无法识别
            if not name or '无法识别' in name or '不知道' in name:
                print('判定：无法识别（名称为空或包含无法识别）')
                return {
                    'success': False,
                    'name': '无法识别',
                    'background': background if background else '模型无法确定图片内容',
                    'confidence': confidence,
                    'message': '无法识别'
                }

            # 判断置信度是否高于 90%
            if confidence > 90:
                print('判定：识别成功')
                return {
                    'success': True,
                    'name': name,
                    'background': background,
                    'confidence': confidence,
                    'message': '识别成功'
                }
            else:
                print(f'判定：置信度过低 ({confidence})')
                return {
                    'success': False,
                    'name': name,
                    'background': background,
                    'confidence': confidence,
                    'message': '无法识别（置信度低于 90%）'
                }
        else:
            return {
                'success': False,
                'message': f'API 调用失败：{response.code} - {response.message}'
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f'识别过程出错：{str(e)}'
        }


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传（HTTP 方式）"""
    if 'file' not in request.files:
        return json.dumps({'success': False, 'message': '没有文件上传'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return json.dumps({'success': False, 'message': '文件名为空'}), 400
    
    if file:
        # 保存文件
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 进行识别
        result = recognize_artifact(filepath)
        
        # 清理上传的文件
        os.remove(filepath)
        
        return json.dumps(result)


@socketio.on('connect')
def handle_connect():
    """处理 WebSocket 连接"""
    print('客户端已连接')
    emit('server_message', {'data': '已连接到服务器'})


@socketio.on('disconnect')
def handle_disconnect():
    """处理 WebSocket 断开"""
    print('客户端已断开连接')


@socketio.on('recognize_image')
def handle_recognize(data):
    """处理 WebSocket 图片识别请求"""
    try:
        image_data = data.get('image')  # base64 编码的图片数据
        
        if not image_data:
            emit('recognition_result', {
                'success': False,
                'message': '未收到图片数据'
            })
            return
        
        # 保存临时文件
        import base64
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        filename = f"temp_{request.sid}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # 进行识别
        result = recognize_artifact(filepath)
        
        # 清理临时文件
        os.remove(filepath)
        
        # 发送结果
        emit('recognition_result', result)
        
    except Exception as e:
        emit('recognition_result', {
            'success': False,
            'message': f'识别失败：{str(e)}'
        })


if __name__ == '__main__':
    # 检查 API 密钥
    if not QWEN_API_KEY:
        print('警告：未设置 DASHSCOPE_API_KEY 环境变量')
        print('请运行以下命令设置 API 密钥:')
        print('  Windows: set DASHSCOPE_API_KEY=your_api_key')
        print('  Linux/Mac: export DASHSCOPE_API_KEY=your_api_key')
    
    print('启动文物/艺术品识别服务器...')
    print('访问地址：http://127.0.0.1:5000')
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
