![[Pasted image 20230420165828.png|500]]
want to get rid of some details, but just use the abstract semantic information to speed up the process. Thus a compression procedure is stacked at the start

+ Latent diffusion uses a separate model called a Variational Auto-Encoder (VAE) to **compress** images to a smaller spatial dimension. Given enough training data, a VAE can hopefully learn to produce a much smaller representation of an input image and then reconstruct the image based on this small **latent** representation with a high degree of fidelity. 
+ The VAE used in SD takes in 3-channel images and produces a 4-channel latent representation with a reduction factor of 8 for each spatial dimension. That is, a **512px** square input image will be compressed down to a **4x64x64 latent**.



