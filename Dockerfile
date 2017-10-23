ARG   BASE_IMAGE="nvidia/cuda:8.0-cudnn6-devel"
FROM $BASE_IMAGE

LABEL maintainer="Mike Voets <mwhg.voets@gmail.com>"

COPY build /tmp/build
WORKDIR /tmp

SHELL ["bash", "-c"]

RUN apt-get update && apt-get install -y \
      ca-certificates \
      curl \
      software-properties-common \
      && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://bootstrap.pypa.io/get-pip.py | python3

# Build dependencies for protobuf.
RUN apt-get update && apt-get install -y \
      autoconf \
      automake \
      build-essential \
      libtool \
      python3-dev \
      unzip \
      && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/google/protobuf/archive/v3.4.0.tar.gz | tar -xz && \
    cd protobuf* && \
    ./autogen.sh && \
    ./configure && \
    make install -j$(nproc) && \
    cd python && \
    python3 setup.py install --cpp_implementation

# Build dependencies for TensorFlow.
RUN echo "deb http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y \
      openjdk-8-jdk \
      python \
      && \
    curl -O -L https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-without-jdk-installer-linux-x86_64.sh && \
    chmod +x bazel-0.5.2-without-jdk-installer-linux-x86_64.sh && \
    ./bazel-0.5.2-without-jdk-installer-linux-x86_64.sh && \
    rm bazel-0.5.2-without-jdk-installer-linux-x86_64.sh && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/tensorflow/tensorflow/archive/v1.3.0.tar.gz | tar -xz && \
    cd tensorflow* && \
    source /tmp/build/tensorflow-env && \
    # Nifty hack!
    sed -ri "/^\W+sha256 = \"[^\"]+\"\W+$/d" tensorflow/workspace.bzl && \
    pip3 install --no-cache-dir numpy && \
    ./configure && \
    bazel build \
      --config=opt \
      # Build with CUDA support when using a CUDA base image.
      ${CUDA_VERSION:+--config=cuda} \
      tensorflow/tools/pip_package/build_pip_package && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD && \
    pip3 install "$(ls -1 tensorflow*.whl | head -n 1)"

RUN ln -s /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda-8.0/lib64/libcuda.so.1

RUN apt-get update && apt-get install -y \
      libglib2.0-0 \
      libgtk2.0-dev \
      python3-tk \
      libsm6 \
      && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
      keras \
      opencv-python>=3.3.0 \
      numpy \
      scipy \
      hyperopt \
      matplotlib \
      sklearn \
      pandas \
      h5py \
      Pillow \
      networkx==1.11

RUN apt-get purge --autoremove -y \
      autoconf \
      automake \
      ca-certificates \
      curl \
      libtool \
      openjdk-8-jdk \
      python \
      python3-dev \
      software-properties-common \
      unzip \
      && \
    rm -rf /etc/apt/sources.list.d/*

EXPOSE 6006

RUN rm -rf /tmp/*

RUN mkdir -p /retinalearn

COPY . /retinalearn

WORKDIR /retinalearn

CMD ["python3", "-u", "retina2.py"]
