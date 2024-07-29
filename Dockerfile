# 使用官方的 Python 3.8 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装 Python 依赖项
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 复制项目文件到工作目录
COPY ./pth /app/pth
COPY ./server.py /app/app.py

# 暴露端口
EXPOSE 8006

# 运行 Flask 应用
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8006", "app:app"]

