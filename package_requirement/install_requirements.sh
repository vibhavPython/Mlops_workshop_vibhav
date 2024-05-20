#!/bin/bash
python --version
pip install --upgrade azure-cli --default-timeout=100
pip install --upgrade azureml-sdk
pip install -r requirements.txt