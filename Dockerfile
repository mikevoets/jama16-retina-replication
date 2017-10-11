ARG   BASE_IMAGE="gcr.io/tensorflow/tensorflow:latest-gpu"
FROM $BASE_IMAGE

LABEL maintainer="Mike Voets <mwhg.voets@gmail.com>"

COPY build /tmp/build
WORKDIR /tmp

SHELL ["bash", "-c"]

RUN apt-get update && apt-get install -y \
      ca-certificates \
      curl \
      software-properties-common \
      libglib2.0-0 \
      libgtk2.0-dev \
      python3-tk \
      libsm6 \
      && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://bootstrap.pypa.io/get-pip.py | python3

RUN pip3 install --no-cache-dir \
      keras \
      networkx==1.11 \
      opencv-python>=3.3.0 \
      numpy \
      scipy \
      hyperopt \
      matplotlib \
      sklearn \
      pandas \
      h5py \
      Pillow

EXPOSE 6006

RUN rm -rf /tmp/*

RUN mkdir -p /retinalearn

COPY . /retinalearn

WORKDIR /retinalearn

CMD ["python3", "-u", "retina2.py"]
