FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 
ENV cudnn_version=8.9.7.*
ENV cuda_version=cuda12.2

WORKDIR /app

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

RUN apt-get update -y
#RUN apt-get install \
#	libcudnn8=${cudnn_version}-1+${cuda_version} \
#	libcudnn8-dev=${cudnn_version}-1+${cuda_version} \
#	libcudnn8-samples=${cudnn_version}-1+${cuda_version}

RUN apt-get install python3-pip ffmpeg zlib1g libsm6 libxext6 -y


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY additional_requirements.txt additional_requirements.txt
RUN pip install -r additional_requirements.txt --no-cache-dir

COPY . .
EXPOSE 8888

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["streamlit", "run", "Main.py"]