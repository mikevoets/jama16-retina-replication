ARG   BASE_IMAGE="nvidia/cuda:8.0-cudnn6-devel"
FROM $BASE_IMAGE

LABEL maintainer="Mike Voets <mwhg.voets@gmail.com>"

COPY build /tmp/build
WORKDIR /tmp

SHELL ["bash", "-c"]

RUN apt-get update && apt-get install -y \
      build-essential \
      ca-certificates \
      curl \
      python \
      software-properties-common \
      && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
      python3.6 \
      python3.6-dev \
      && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN echo "deb http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install -y \
      bazel \
      openjdk-8-jdk \
      && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
      autoconf \
      automake \
      libtool \
      unzip \
      && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/google/protobuf/archive/v3.4.0.tar.gz | tar -xz && \
    cd protobuf* && \
    ./autogen.sh && \
    ./configure && \
    make install -j$(nproc) && \
    cd python && \
    python3.6 setup.py install --cpp_implementation

RUN pip3.6 install --no-cache-dir \
      numpy

RUN curl -L https://github.com/tensorflow/tensorflow/archive/v1.3.0.tar.gz | tar -xz && \
    cd tensorflow* && \
    # HACK: Temporarily remove SHA256 checksums because reasons.
    # ¯\_(ツ)_/¯ https://github.com/tensorflow/tensorflow/issues/12979
    sed -ri "/^\W+sha256 = \"[^\"]+\"\W+$/d" tensorflow/workspace.bzl && \
    source /tmp/build/tensorflow-env && ./configure && \
    bazel build \
      --config=opt \
      # Build with CUDA support when using a CUDA base image.
      ${CUDA_VERSION:+--config=cuda} \
      tensorflow/tools/pip_package/build_pip_package

RUN cd tensorflow* && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD && \
    ls -1 && \
    pip3.6 install "$(ls -1 tensorflow*.whl | head -n 1)"


RUN pip3.6 install --no-cache-dir \
      bleach>=2.0 \
      html5lib>=0.99999999 \
      keras>=2.0.7 \
      opencv-python>=3.3.0 \
      jupyter \
      numpy \
      scipy \
      Pillow \
      matplotlib \
      sklearn \
      pandas \
      h5py

RUN apt-get purge --autoremove -y \
      autoconf \
      automake \
      bazel \
      curl \
      libtool \
      openjdk-8-jdk \
      python \
      software-properties-common \
      unzip \
      && \
    rm -rf /etc/apt/sources.list.d/*

COPY jupyter_notebook_config.py /root/.jupyter/

COPY run_jupyter.sh /root/

EXPOSE 6006 8888

RUN rm -rf /tmp/*

WORKDIR /

CMD ["/bin/bash"]

RUN apt-get update && apt-get install -y libgtk2.0-dev
