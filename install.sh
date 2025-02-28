conda create -n zo_policy python=3.9 -y
conda activate zo_policy
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers
conda install huggingface_hub -y
pip install accelerate
conda install datasets -y
pip install wandb
conda install scikit-learn -y
pip install openai
pip install modelscope
pip install debugpy
# pip install -U libauc
pip install icecream
# pip install ollama
pip install transformers -U
# conda install pymysql
pip install transformers -U