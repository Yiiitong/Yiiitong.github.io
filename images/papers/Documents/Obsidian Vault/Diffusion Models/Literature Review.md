**1. D2C: Diffusion-Decoding Models for Few-Shot Conditional Generation**
+ a special VAE that is suitable for conditional few-shot generation
+ D2C uses contrastive self-supervised learning methods to obtain a latent space that inherits the transferrability and few-shot capabilities of self-supervised representations.
+ combines a discriminative model providing conditioning signal and generative diffusion model over the latent space
+ *D2C models produce samples by drawing $z^{(1)}$ from a diffusion process and then decoding $x$ from $z^{(1)}$*

**2. +++Diffusion Autoencoder toward a meaningful and decodable representation**
+ Our method can encode any image into a two-part latent code where the first part is semantically meaningful and linear, and the second part captures stochastic details, allowing near-exact reconstruction.
+ With DDIM, it is possible to run the generative process backward deterministically to obtain the noise map $x_T$ , which represents the latent variable or encoding of a given image $x_0$.
+ ![[Pasted image 20230516113758.png|400]]
+ 