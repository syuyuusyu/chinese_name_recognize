# 使用官方的 Python 3.8 基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装 Python 依赖项
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件到工作目录
COPY . .

# 暴露端口
EXPOSE 8006

# 运行 Flask 应用
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8006", "app:app"]