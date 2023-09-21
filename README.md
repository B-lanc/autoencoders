# autoencoders
Comparison between regularized autoencoder vs variational autoencoder vs VQGAN in terms of reconstruction loss (not for generation)
Uses 102flowers dataset
This means the latent space will preferrably be something not much smaller than the input dimensions


# Building the docker image
> docker build . -t b-lanc/autoencoders

# Running the docker container
> docker run -dit --name=autoencoders --runtime=nvidia --gpus=0 --shm-size=16gb -v .:/workspace -v /mnt/Data/datasets/102flowers/jpg:/dataset -v /mnt/Data2/DockerVolumes/autoencoders:/saves b-lanc/autoencoders

# Going into the docker container
> docker exec -it autoencoders /bin/bash
