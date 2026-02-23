cat > Dockerfile <<'EOF'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# 基本ツール
RUN apt-get update && apt-get install -y \
    locales curl gnupg2 lsb-release \
    build-essential cmake git \
    python3-pip python3-rosdep python3-vcstool python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ROS2 apt repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    && rm -rf /var/lib/apt/lists/*

# rosdep 初期化（すでにされていてもOK）
RUN rosdep init || true
RUN rosdep update

# --- ROS1 (Noetic相当) を "upstream" で必要最小限ビルド ---
# ros_comm + 共通メッセージぐらいを入れる（カスタムmsgなし前提）
RUN mkdir -p /opt/ros1_ws/src
WORKDIR /opt/ros1_ws

RUN pip3 install -U rosinstall_generator

# ros_comm と common_msgs を取得
RUN rosinstall_generator ros_comm common_msgs --rosdistro noetic --deps --tar \
    > ros1.rosinstall && \
    vcs import src < ros1.rosinstall

# 依存解決（Jammy上でNoeticソースをビルドするため、ここが一番コケやすい箇所）
RUN apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y \
      --rosdistro noetic && \
    rm -rf /var/lib/apt/lists/*

# ビルド（catkin_make_isolated）
RUN ./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release

# --- ros1_bridge をソースビルド ---
RUN mkdir -p /opt/bridge_ws/src
WORKDIR /opt/bridge_ws

RUN git clone https://github.com/ros2/ros1_bridge.git src/ros1_bridge

# ブリッジの依存
RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y && rm -rf /var/lib/apt/lists/*

# ROS1 + ROS2 の環境を両方sourceしてビルド
RUN source /opt/ros/humble/setup.bash && \
    source /opt/ros1_ws/install_isolated/setup.bash && \
    colcon build --symlink-install

# 起動時に使いやすいように
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /opt/ros1_ws/install_isolated/setup.bash" >> /root/.bashrc && \
    echo "source /opt/bridge_ws/install/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
EOF
