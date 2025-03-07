FROM fizzzzz/uv-cuda:uv0524-py310-cuda124-ubuntu2204

# Install vim and git, then clone the IntroGPU repo using /bin/sh
RUN apt-get update && \
    apt-get install -y vim git && \
    git clone https://github.com/klxu03/IntroGPU.git $HOME/IntroGPU && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
