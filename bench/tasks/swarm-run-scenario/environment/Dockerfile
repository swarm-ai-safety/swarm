FROM python:3.12-slim
WORKDIR /root

# Install swarm framework
COPY swarm-package/ /root/swarm-package/
RUN pip install --no-cache-dir /root/swarm-package/

# Copy scenarios
COPY scenarios/ /root/scenarios/

# Create output directory
RUN mkdir -p /root/output
