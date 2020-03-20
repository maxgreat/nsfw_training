FROM pytorch/pytorch:latest

#RUN apt-get update && apt-get install -y --no-install-recommends \
ENV http_proxy=http://10.100.9.1:2001 https_proxy=http://10.100.9.1:2001

RUN conda install -c conda-forge -c pytorch -y tensorflow seaborn Cython scikit-learn nodejs matplotlib jupyter jupyterlab tensorboardx ignite

# RUN jupyter labextension install jupyterlab_tensorboard

RUN pip install jupyter_tensorboard torchvision


WORKDIR /workspace

COPY entrypoint.sh /

ENTRYPOINT [ "/entrypoint.sh" ]
