#!/bin/bash
set -a

# Append env to .profile
echo -e '\n' >> /home/jovyan/.bashrc
echo -e 'export http_proxy=http://wwwproxy.uni-muenster.de:3128' >> /home/jovyan/.bashrc
echo -e 'export https_proxy=http://wwwproxy.uni-muenster.de:3128' >> /home/jovyan/.bashrc
echo -e 'alias ll="ls -al"' >> /home/jovyan/.bashrc

http_proxy=http://wwwproxy.uni-muenster.de:3128
https_proxy=http://wwwproxy.uni-muenster.de:3128
JUPYTER_CONFIG_DIR=$JUP_HOME/.jupyter

if [ ! -d "$JUP_HOME" ]; then
  mkdir -p $JUP_HOME
fi

if [ ! -e "$JUPYTER_CONFIG_DIR/jupyter_lab_config.py" ]; then
  jupyter lab --generate-config
fi

jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --NotebookApp.notebook_dir=$JUP_HOME
