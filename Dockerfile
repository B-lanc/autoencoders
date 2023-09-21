FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN pip install --upgrade \
  pip \
  lightning==2.0.9 \
  matplotlib==3.8.0 \
  albumentations==1.3.1