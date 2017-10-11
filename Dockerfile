ARG   BASE_IMAGE="gcr.io/tensorflow/tensorflow:latest-gpu-py3"
FROM $BASE_IMAGE

LABEL maintainer="Mike Voets <mwhg.voets@gmail.com>"

COPY build /tmp/build
WORKDIR /tmp

SHELL ["bash", "-c"]

# Install dependencies.
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

# Get pip3.
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# Install runtime (pip3) dependencies.
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
