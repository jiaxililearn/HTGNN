# setting env
sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda
pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html

# cuda version DGL
pip install  dgl -f https://data.dgl.ai/wheels/cu102/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html