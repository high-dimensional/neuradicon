FROM nvcr.io/nvidia/cuda:12.2.2-runtime-ubuntu22.04
LABEL author="Henry Watkins"
LABEL author_contact="h.watkins@ucl.ac.uk"
LABEL description="neuradicon dashboard docker image for the deployment of the neuradicon neuroradiological reporting tool"
LABEL version="1.0"

RUN apt update
RUN apt install python3.10 python3-pip -y

WORKDIR /app

COPY ./dist /app/dist
COPY ./model_dir /app/model_dir
COPY ./examples /app/examples

RUN pip install dist/neuradicon-0.0.1-py3-none-any.whl

