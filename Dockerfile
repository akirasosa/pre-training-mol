FROM akirasosa/ubuntu:0.2.0

RUN mkdir -p ~/data/
RUN curl -sSL "https://www.dropbox.com/s/fifvs2gpdnocxxr/qm9.parquet?dl=1" > ~/data/qm9.parquet

RUN pip install -U pip \
  && pip install -U \
  dacite \
  matplotlib \
  numpy \
  omegaconf \
  pandas \
  pyarrow \
  pytorch-lightning \
  scikit-learn \
  scipy \
  sympy \
  tensorboard \
  timm \
  torch \
  torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  torch_optimizer \
  torchvision \
  && rm -rf ~/.cache/pip

USER root

