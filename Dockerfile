FROM pytorch/pytorch:latest


RUN conda install -c conda-forge -c pytorch -y tensorflow seaborn Cython scikit-learn nodejs matplotlib jupyter jupyterlab tensorboardx ignite


RUN pip install jupyter_tensorboard torchvision


WORKDIR /workspace

COPY entrypoint.sh /

ENTRYPOINT [ "/entrypoint.sh" ]
