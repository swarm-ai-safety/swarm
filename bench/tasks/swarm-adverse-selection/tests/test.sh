#!/bin/bash
set -e
cd /root
python3 -m pytest tests/test_outputs.py -v
