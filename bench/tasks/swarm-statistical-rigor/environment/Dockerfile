FROM python:3.12-slim
WORKDIR /root

RUN pip install --no-cache-dir scipy pandas numpy

COPY data/ /root/data/

RUN mkdir -p /root/output
