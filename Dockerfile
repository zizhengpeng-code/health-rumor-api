# 微信云托管镜像（D53）。知识库不打进镜像，启动时从对象存储下载（见 server.py）。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app
COPY requirements.txt .
RUN pip install -i https://mirrors.cloud.tencent.com/pypi/simple -r requirements.txt

COPY . .

EXPOSE 80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
