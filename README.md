Guide to open Jupyter Notebook on a remote server.
https://web.archive.org/web/20200628012208/https://docs.anaconda.com/anaconda/user-guide/tasks/remote-jupyter-notebook

Check CUDA and CUDA Toolkit----  \
nvidia-smi  \
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"  \
which nvcc && nvcc --version  --> if no toolkit--> conda install -c nvidia cuda-toolkit=<cuda_version> -y  \
**If that doesn't work, use NVIDIA official site instructions**

Set environement variables after toolkit installation----  \
Check if available: ls -la /usr/local/cuda/include/ | grep bf16  \
export CUDA_HOME=/usr/local/cuda  \
export PATH=$CUDA_HOME/bin:$PATH  \
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH  \
export CPATH=$CUDA_HOME/include:$CPATH

Tensorboard command---- \
In remote terminal 2 --> \
... install *tensorboard* and *tensorflow*: pip install tensorflow tensorboard \
... initiate tensorboard: tensorboard --logdir=runs/rwkv_separator --port=6007 \
In local terminal --> ssh -p 43381 root@209.226.130.26 -L 6007:localhost:6007 (change this per the remote server)\
In local browser --> http://localhost:6007

Also used this to write and inspect a log---- \
python train_separator_fixed.py 2>&1 | tee training.log \
tail -f training.log *or* tail -f training.log | grep -E "Epoch|val_loss|Saved"
