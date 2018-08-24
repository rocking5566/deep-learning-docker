ARG tf_ver=1.8.0
FROM tensorflow_object_detection:tf_${tf_ver}

RUN pip install keras

# [Warning] - Do not use root in public IP 

CMD jupyter notebook --allow-root --port=8888 --ip=0.0.0.0 & \
    tensorboard --logdir=/log/tensorboard
