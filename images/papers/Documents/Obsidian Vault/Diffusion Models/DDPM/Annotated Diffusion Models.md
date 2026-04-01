[Annotated Diffusion Models](https://huggingface.co/blog/annotated-diffusion)
forward diffusion process: ![[Pasted image 20230418144349.png]]
Basically, each new (slightly noisier) image at time step $t$ is drawn from a **conditional Gaussian distribution** with $\mu_t = \sqrt{1-\beta_t}~x_{t-1}$ amd $\sigma_t^2 = \beta_t$, which we can do by sampling $\epsilon \sim \mathcal{N}(0,I)$ and then setting $x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon$ 

-> *Since $\beta_t$ is not constant at each time step, we need to define a 'variance schedule'  (can be lineaar, quadratic, cosine...), similar like a learning rate schedule.*

One nice property: ![[Pasted image 20230421104551.png|300]]
$\alpha_t​:=1−\beta_t$​ and $\bar\alpha_t​:=\Pi_{s=1}^t​\alpha_s​$.
-> so we can sample Gaussian noise and scale it appropriatly and add it to $x_0​$ to get $x_t$ directly.

To run the reverse process, we need to know the conditional distribution $p(x_{t-1}|x_t)$, but it's intractable since it requires knowing the distribution of all possible images in order to calculate. 
-> Hence, we're going to leverage a neural network to **approximate (learn) this conditional probability distribution**, let's call it $p_\theta(x_{t-1}|x_t)$, with $\theta$ being the parameters of the neural network, updated by gradient descent.

-> we also assume the reverse process to be Gaussian, so we parametrize it as:
$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta (x_t, t), \Sigma_\theta(x_t, t))$
in which $t$ is the noise level
-> *DDPM authors decided to **keep the variance fixed, and let the neural network only learn (represent) the mean $\mu_\theta$​ of this conditional probability distribution**.*
-> This was then later improved in the [Improved diffusion models](https://openreview.net/pdf?id=-NEXDKk8gZ) paper, where a neural network also learns the variance of this backwards process, besides the mean.

To derive an objective function to learn the mean of the backward process, the authors observe that the combination of $q$ and $p_θ$​ can be seen as a variational auto-encoder (VAE). 
-> Hence, the **variational lower bound** (also called ELBO) can be used to minimize the negative log-likelihood with respect to ground truth data sample $x_0$.
-> turns out that the ​ELBO for this process is a sum of losses at each time step $t$, $L = L_0 + L_1 +...+ L_T​$. 

-> *By construction of the forward $q$ process and backward process, each term (except for $L_0$) of the loss is actually the **KL divergence between 2 Gaussian distributions** which **can be written explicitly as an L2-loss with respect to the means!!***

Because of this and the nice property, during training, we can **optimize random terms of the loss function L** (or in other words, to randomly sample $t$ during training and optimize $L_t$​).

Furthermore, we can **reparametrize the mean to make the neural network learn (predict) the added noise (via a network $\epsilon_\theta(x_t​,t)$) for noise level t** in the KL terms which constitute the losses. 
-> *our neural network becomes a noise predictor, rather than a (direct) mean predictor. The mean can be computed as follows*:
![[Pasted image 20230421110134.png|300]]
The final objective function $L_t$​ then looks as follows (for a random time step t, $\epsilon \sim \mathcal{N}(0,I)$ is the pure noise sampled at time step t):
![[Pasted image 20230421110219.png|400]]
in which $\epsilon_\theta(x_t​,t)$ is our neural network. The neural network is optimized using a simple mean squared error (MSE) between the true and the predicted Gaussian noise.

Training steps:
![[Pasted image 20230421110451.png|350]]
Sampling steps: (generating new images)
![[Pasted image 20230421121837.png|350]]

### Implementation details:
#### Neural network
The neural network needs to take in a noised image at a particular time step and return the predicted noise. (U-Net)
-> the predicted noise is a tensor that has the same size as the input image.
#### Position embedding
As the parameters of the neural network are shared across time (noise level), the authors employ sinusoidal position embeddings to encode $t$, inspired by the Transformer
-> *This makes the neural network "know" at which particular time step (noise level) it is operating, for every image in a batch.*

-> (The `SinusoidalPositionEmbeddings` module takes a tensor of shape `(batch_size, 1)` as input (i.e. the noise levels of several noisy images in a batch), and turns this into a tensor of shape `(batch_size, dim)`, with `dim` being the dimensionality of the position embeddings.)
#### ResNet block
DDPM authors employed a Wide ResNet block ([Zagoruyko et al., 2016](https://arxiv.org/abs/1605.07146))
Phil Wang has replaced the standard convolutional layer by a *"weight standardized"* version, which works better in combination with *group normalization* ([Kolesnikov et al., 2019](https://arxiv.org/abs/1912.11370))
#### Attention module
added in between the convolutional blocks (regular multi-head self-attention (as used in the Transformer))




