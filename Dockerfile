FROM pytorch/pytorch:lastest

#RUN apt-get update && apt-get install -y --no-install-recommends \

RUN conda install -y scikit-learn matplotlib jupyter jupyterlab tensorboardx

RUN jupyter labextension install jupyterlab_tensorboard

RUN pip install jupyter_tensorboard torchvision

RUN mkdir -p /home/me && chmod 1777 /home/me

ENV HOME /home/me

EXPOSE 6006
EXPOSE 8888

COPY entrypoint.sh /

ENTRYPOINT [ "/entrypoint.sh" ]