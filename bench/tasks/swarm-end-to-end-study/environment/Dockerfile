FROM python:3.12-slim
WORKDIR /root

COPY swarm-package/ /root/swarm-package/
RUN pip install --no-cache-dir /root/swarm-package/ scipy matplotlib

COPY scenarios/ /root/scenarios/

RUN mkdir -p /root/output/{sweep,analysis,plots,paper}
