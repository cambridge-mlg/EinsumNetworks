#!/bin/bash

rm -rf venv/
virtualenv --system-site-packages -p python3.6 ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
cd src
python datasets.py
