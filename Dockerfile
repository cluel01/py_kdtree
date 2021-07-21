FROM python:3.8

RUN apt-get update
RUN apt-get install -y git

#Jupyter
RUN groupadd -r jovyan -g 1000 && useradd -s /bin/bash -m -u 1000 -r -g jovyan jovyan
ENV JUP_HOME /home/jovyan/work
WORKDIR /home/jovyan
RUN pip install notebook jupyterlab
COPY --chown=jovyan start.sh .
RUN chown -R 1000:1000 /home/jovyan
USER jovyan
RUN mkdir /home/jovyan/work

RUN chmod u+x /home/jovyan/start.sh

#App
WORKDIR /home/jovyan/work
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt
COPY --chown=jovyan setup.py README.rst py_kdtree ./
RUN pip install -e .

CMD ["/home/jovyan/start.sh"]
