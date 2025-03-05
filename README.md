# CS4248
Emoji Transformer

# About
This project aims to create a transformer tht can compress the representation of English into emojis.

# Installation
- This project uses Python 3.10, and torch in order to train and run inference on the transformer.

### For GPU chads
1) If you have uv installed, then just run `uv sync --extra cu124` 

<sub>Note: (This should download all dependencies for cuda 12.4) If your cuda version is different please let the authors know.</sub>

2) If you do not use uv, run `pip install -r requirements_gpu.txt` 

### For CPU Sufferers
1) If you have uv installed, just run `uv sync --extra cpu`
2) If you do not use uv, run `pip install -r requirements_cpu.txt`


# Team
Yao Hejun
Luca Baumann
Jin Chun
Zhang Mengjie
Ma Jia Jun
Warren Low
