Diffusion models are a family of probabilistic generative models that progressively destruct data by injecting noise, then learn to reverse this process for sample generation.

## Foundations
### Denoising Diffusion Probabilistic Models (DDPMs)
+ two Markov chains -> 
+ a forward chain that perturbs data to noise (hand-designed with the goal to transform any data distribution into a simple prior distribution (e.g., standard Gaussian))
+ a reverse chain that converts noise back to data (reverses the former by learning transition kernels parameterized by deep neural networks)

**Forward process:**
transition kernel $q(x_t | x_{t-1})$
joint distribution of $x_1$,... $x_T$ conditioned on $x_0$: $q(x_1, ..., x_T|x_0)$
it means that we apply $q$ repeatedly from timestep $1$ to $T$. It's also called trajectory.
-> $q(x_1, ..., x_T|x_0) = \Pi_{t=1}^T~q(x_t|x_{t-1})$
typical design for the transition kernel is Gaussian perturbation:![[Pasted image 20230424134053.png|300]]
This Gaussian transition kernel allows us to marginalize the joint distribution to obtain the analytical form of $q(x_t | x_0)$ for all t: ![[Pasted image 20230421104551.png|300]]
$\alpha_t​:=1−\beta_t$​ and $\bar\alpha_t​:=\Pi_{s=1}^t​\alpha_s​$.
-> Given $x_0$, we can easily obtain a sample of $x_𝑡$ by sampling a Gaussian vector $\epsilon \sim \mathcal{N}(0,I)$ and applying the transformation:
$x_t = \sqrt{\bar{\alpha_t}}x_{0} + \sqrt{1-\bar{\alpha_t}}\epsilon$ 

when $\bar\alpha_t​ \approx 0$, $x_T$ is almost Gaussian in distribution, so we have:
$q(x_T) = \int q(x_T|x_0)q(x_0)dx_0 \approx \mathcal{N}(x_T; 0,I)$

**Reverse process:**
For generating new data samples, DDPMs start by first generating an unstructured noise vector from the prior distribution, then gradually remove noise therein by running a learnable Markov chain in the reverse time direction.

the reverse Markov chain is parameterized by a prior distribution $p(x_T) = \mathcal{N}(x_T; 0,I)$ and a learnable transition kernel $p_\theta(x_{t-1}|x_t)$:
$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta (x_t, t), \Sigma_\theta(x_t, t))$
in which $\theta$ denotes model parameters and mean and variance are parameterized by deep neural networks.
--> With this reverse Markov chain in hand, we can generate a data sample $x_0$ by first sampling a noise vector $x_𝑇 ∼ 𝑝(x_𝑇 )$, then iteratively sampling from the learnable transition kernel $x_{𝑡−1} ∼ 𝑝_𝜃 (x_{𝑡−1} | x_𝑡)$ until $𝑡 = 1$.

*Key to the success of the sampling process*: 
training the reverse Markov chain to match the actual time reversal of the forward Markov chain
-> adjust parameter $\theta$ so that the joint distribution of the reverse Markov chain: $p_\theta(x_0, x_1, ..., x_T) = p(x_T) \Pi_{t=1}^T~p_\theta(x_{t-1}|x_t)$
closely approximates that of the forward process :
$q(x_0, x_1, ..., x_T) = q(x_0) \Pi_{t=1}^T~q(x_t|x_{t-1})$
-> *This is achieved by minimizing the Kullback-Leibler (KL) divergence between these two:*
$KL(q(x_0, x_1,...,x_T) || p_\theta(x_0, x_1,...,x_T))$
$= -\mathbb{E}_{q(x_0, x_1,...,x_T)}[log~p_\theta(x_0, x_1,...,x_T)] + const$ **(definition of KL divergence)
$= \mathbb{E}_{q(x_0, x_1,...,x_T)} [-log~p(x_T) - \sum_{t=1}^T log~\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}] + const$ (q(..) and p(...) are both products of distribution)
$\ge \mathbb{E}[-log~ p_\theta(x_0)] + const$ (Jensen’s inequality)

Especially:
$𝐿_{VLB} = \mathbb{E}_{q(x_0, x_1,...,x_T)} [-log~p(x_T) - \sum_{t=1}^T log~\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}]$ is the variational lower bound (VLB) of the log-likelihood of the data $x_0$, a common objective for training probabilistic generative models. 
--> *The objective of DDPM training is to maximize the VLB (or equivalently, minimizing the negative VLB)*, which is particularly easy to optimize because it is a sum of independent terms, and can thus be estimated efficiently by Monte Carlo sampling and optimized effectively by stochastic optimization.

Ho et al. (2020) propose to reweight various terms in $𝐿_{VLB}$ for better sample quality and noticed an important equivalence between the resulting loss function and the training objective for noise-conditional score networks (NCSNs), one type of score-based generative models, in Song and Ermon. The loss is:
$\mathbb{E}_{t \sim\mathcal{U}[1,T], x_i \sim q(x_0), \epsilon \sim \mathcal{N}(0,I)} [\lambda (t) \|\epsilon - \epsilon_\theta(x_t, t)\|^2]$
in which $\lambda (t)$ is a positive weighting function, $\epsilon_\theta$ is a deep neural network with parameter 𝜃 that predicts the noise vector 𝝐 given $x_𝑡$ and $t$. 
--> this loss reduces to $𝐿_{VLB}$ for a particular choice of the weighting function $\lambda (t)$, and *has the same form as the loss of denoising score matching over multiple noise scales for training score-based generative models.*

### Score-Based Generative Models (SGMs)
The core of score-based generative models: (Stein) score (a.k.a., score or score function):
Given a probability density function $𝑝(x)$, its score function is defined as the gradient of the log probability density $∇_x log ~𝑝(x)$.
-> Unlike the commonly used Fisher score $∇_𝜃 log~ 𝑝_𝜃 (x)$ in statistics, the Stein score considered here is *a function of the data $x$ rather than the model parameter 𝜃. It is a vector field that points to directions along which the probability density function has the largest growth rate.

**Key idea:** perturb data with a sequence of *intensifying Gaussian noise* and jointly *estimate the score functions for all noisy data distributions* by training a deep neural network model conditioned on noise levels (called a noise-conditional score network, NCSN)

+ Samples are generated by chaining the score functions at decreasing noise levels with score-based sampling approaches, including ==Langevin Monte Carlo , stochastic differential equations, ordinary differential equations==, and their various combinations.
+ Training and sampling are completely ==decoupled== in the formulation of score-based generative models, so one can use a multitude of sampling techniques after the estimation of score functions.

let $q(x_0)$ be the data distribution, $0<\sigma_1 < \sigma_1 <...<\sigma_T$ be a sequence of noise levels.
A typical example of SGMs involves perturbing a data point $x_0$ to $x_𝑡$ by the Gaussian noise distribution $q(x_t|x_0) = \mathcal{N} (x_t; x_0, \sigma_t^2 I)$. This yields a sequence of noisy data densities $q(x_1)$, ..., $q(x_T)$, where $q(x_t) = \int q(x_t)q(x_0)dx_0$. 
-> *A noise-conditional score network is a deep neural network $s_𝜃 (x, 𝑡)$ trained to estimate the score function $∇_{x_𝑡} log~𝑞(x_𝑡)$.

+ Learning score functions from data (a.k.a., score estimate) has established techniques such as score matching, denoising score matching, and sliced score matching, so we can directly employ one of them to train our noise-conditional score networks from perturbed data points.
+ For example, with denoising score matching, the training objective is given by:
$\mathbb{E}_{t \sim\mathcal{U}[1,T], x_0 \sim q(x_0), x_t \sim q(x_t | x_0)} [\lambda (t) ~\sigma_t^2 \|∇_{x_𝑡} log~𝑞(x_𝑡) - s_𝜃 (x_t, 𝑡)\|^2]$
$= \mathbb{E}_{t \sim\mathcal{U}[1,T], x_0 \sim q(x_0), x_t \sim q(x_t | x_0)} [\lambda (t) ~\sigma_t^2 \|∇_{x_𝑡} log~𝑞(x_𝑡 |x_0) - s_𝜃 (x_t, 𝑡)\|^2] + const$
$= \mathbb{E}_{t \sim\mathcal{U}[1,T], x_0 \sim q(x_0), x_t \sim q(x_t | x_0)} [\lambda (t) ~\|-\frac{x_t-x_0}{\sigma_t} - \sigma_t s_𝜃 (x_t, 𝑡)\|^2] + const$ 
**(because $q(x_t|x_0) = \mathcal{N} (x_t; x_0, \sigma_t^2 I)$)**
$= \mathbb{E}_{t \sim\mathcal{U}[1,T], x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0,I)} [\lambda (t) ~\|\epsilon + \sigma_t s_𝜃 (x_t, 𝑡)\|^2] + const$
**(because $x_t = x_0 + \sigma_t \epsilon$)**
($\lambda (t)$ is a positive weighting function.)

*-> once we set $\epsilon_\theta(x_t, t) = - \sigma_t s_𝜃 (x_t, 𝑡)$, the training objectives of DDPMs and SGMs are equivalent.*

+ Moreover, one can generalize the score matching with higher order. High-order derivatives of data density provide additional local information about the data distribution.

**For sample generation**, SGMs leverage iterative approaches to produce samples from $s_𝜃 (x,𝑇 ), s_𝜃 (x,𝑇 −1), · · · , s_𝜃 (x, 0)$ in succession. Many sampling approaches exist due to the decoupling of training and inference in SGMs.
-> the first sampling method for SGMs is **annealed Langevin dynamics (ALD)**:
+ 𝑁 : number of iterations per time step, $𝑠_𝑡$ > 0: step size. We first initialize ALD with $x^{(𝑁 )}_𝑇 ∼ \mathcal{N}(0, I)$, then apply Langevin Monte Carlo for $𝑡 = 𝑇 ,𝑇 − 1, · · · , 1$ one after the other. 
+ At each time step $0 ≤ 𝑡 < 𝑇$ , we start with $x^{(0)}_𝑡 = x^{(𝑁)}_{𝑡+1}$ , before iterating according to the following update rule for $𝑖 = 0, 1, · · · , 𝑁 − 1$:
  ![[Pasted image 20230424164213.png|300]]
  + The theory of **Langevin Monte Carlo** guarantees that *as $𝑠_𝑡 → 0$ and $𝑁 → ∞$, $x ^{(𝑁 )}_0$ becomes a valid sample from the data distribution $𝑞(x_0)$.*

### Stochastic Differential Equations (Score SDEs)
DDPMs and SGMs can be further generalized to the case of *infinite time steps or noise levels*, where the perturbation and denoising processes are ==solutions to stochastic differential equations (SDEs)==.
--> We call this formulation **Score SDE**, as it leverages SDEs for noise perturbation and sample generation, and the denoising process requires estimating *score functions of noisy data distributions*.

	To make sense of why we use an SDE, here is a tip: the SDE is inspired by the Brownian motion, in which a number of particles move randomly inside a medium. This randomness of the particles' motion models the continuous noise perturbations on the data.

+ Score SDEs perturb data to noise with a diffusion process governed by the following stochastic differential equation (SDE):
    $dx = f(x,t)dt + g(t)dw$
  in which $f(x,t)$ and $g(t)$ are diffusion and drift functions of the SDE, and w is a standard Wiener process (a.k.a., Brownian motion).

	*You can understand an SDE as a stochastic generalization to ordinary differential equations (ODEs). Particles moving according to an SDE not only follows the deterministic drift $f(x,t)$, but are also affected by the random noise coming from $g(t)dw$.*


  --> The *forward* processes in DDPMs and SGMs are both ==discretizations== of this SDE:
  - for DDPMs, the corresponding SDE is:
     $dx = -\frac{1}{2} \beta(t)x~dt + \sqrt{\beta(t)}dw$
     where $𝛽 ( \frac{t}{T} ) = 𝑇 𝛽_𝑡$ as 𝑇 goes to infinity
  - for SGMs, the corresponding SDE is given by
     $dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}}dw$
     where $\sigma ( \frac{t}{T} ) = \sigma_𝑡$ as 𝑇 goes to infinity

Crucially, for any diffusion process in the form of $dx = f(x,t)dt + g(t)dw$, Anderson shows that it can be reversed by solving the following ==reverse-time SDE==:
![[Pasted image 20230424170030.png|400]]
where $\bar w$ is a standard Wiener process (Brownian motion) when time flows *backwards*, and $d𝑡$ denotes an *infinitesimal negative time step*.
***-> This reverse SDE can be computed once we know the drift and diffusion coefficients of the forward SDE, as well as the score of $p_t(x)$ for each $t∈[0,T]$.***
--> The solution trajectories of this reverse SDE share the same marginal densities as those of the forward SDE, except that they evolve in the opposite time direction.
--> Intuitively, solutions to the reverse-time SDE are diffusion processes that gradually convert noise to data.

![[Pasted image 20230425133311.png]]



Moreover, Song et al. (2020) prove the existence of an **ordinary differential equation (ODE)**, namely the **probability flow ODE**, whose trajectories have the same marginals as the reverse-time SDE. The probability flow ODE is given by:
![[Pasted image 20230424171632.png|300]]
Both the reverse-time SDE and the probability flow ODE allow sampling from the same data distribution as their trajectories have the same marginals.

+ *Once the score function at each time step $t$, $∇_x log~𝑞_𝑡 (x)$, is known, we unlock both the reverse-time SDE and the probability flow ODE and can subsequently generate samples by solving them with various numerical techniques*, such as annealed Langevin dynamics, numerical SDE solvers, numerical ODE solvers, and predictor-corrector methods (combination of MCMC and numerical ODE/SDE solvers)
+ Like in SGMs, we parameterize a time-dependent score model $s_𝜃 (x_𝑡 , 𝑡)$ to estimate the score function by generalizing the score matching objective to continuous time, leading to the following objective:
![[Pasted image 20230424172319.png|500]]

In Summary:
***we can use the time-dependent score function $∇_xlog~p_t(x)$ to construct the reverse-time SDE, and then solve it numerically to obtain samples from $p_0$ using samples from a prior distribution $p_T$. We can train a time-dependent score-based model $s_θ(x,t)$ to approximate $∇_xlog~p_t(x)$, using the following weighted sum of denoising score matching objectives:***
![[Pasted image 20230425133741.png]]
where $\mathcal{U}(0,T)$ is a uniform distribution over $[0,T]$, $p_{0t}(x(t)∣x(0))$ denotes the transition probability from $x(0)$ to $x(t)$, and $λ(t)∈R>0$ denotes a positive weighting function.

***In the objective, the expectation over $x(0)$ can be estimated with empirical means over data samples from $p_0$. The expectation over $x(t)$ can be estimated by sampling from $p_{0t}(x(t)∣x(0))$, which is efficient when the drift coefficient $f(x,t)$ is affine. The weight function $λ(t)$ is typically chosen to be inverse proportional to $E[∥∇xlog~p_{0t}(x(t)∣x(0))∥_2^2]$.***


--> Subsequent research on diffusion models focuses on improving these classical approaches (DDPMs, SGMs, and Score SDEs) from three major directions
+ faster and more efficient sampling
+ more accurate likelihood and density estimation
+ handling data with special structures (such as permutation invariance, manifold structures, and discrete data)

### **DIFFUSION MODELS WITH EFFICIENT SAMPLING**
###### Learning-Free Sampling
+ SDE Slovers
+ ODE Solvers
###### Learning-Based Sampling
+ Optimized Discretization
+ Knowledge Distillation
+ Truncated Diffusion

### **Improved Likelihood**
+ Noise Schedule Optimization
+ Reverse Variance Learning
+ Exact Likelihood Computation

### **Data with Special Structures**
+ Manifold Structures
	*According to the manifold hypothesis, most natural data lies on manifolds with significantly reduced intrinsic dimensionality. Consequently, identifying these manifolds and training diffusion models directly on them can be advantageous due to the lower data dimensionality. Many recent works have built on this idea, starting by using an autoencoder to condense the data into a lower dimensional manifold, followed by training diffusion models in this latent space. In these cases, the manifold is implicitly defined by the autoencoder and learned through the reconstruction loss. In order to be successful, it is crucial to design a loss function that allows for the joint training of the autoencoder and the diffusion models*
	+ examples: Latent Score-Based Generative Model (LSGM), Latent Diffusion Model (LDM)
+ Data with Invariant Structures 
+ Discrete Data

## APPLICATIONS OF DIFFUSION MODELS
### Conditional diffusion models
**Conditioning Mechanisms in Diffusion Models**: 
+ ***concatenation***: diffusion models concatenate informative guidance with intermediate denoised targets in diffusion process, such as label embedding and semantic feature maps.
+ ***gradient-based***: incorporates task-related gradient into the diffusion sampling process for controllable generation: 
	For example, in image generation, one can train an auxiliary classifier on noisy images, and then use gradients to guide the diffusion sampling process towards an arbitrary class label
+ ***cross-attention***: performs attentional message passing between the guidance and diffusion targets, which is usually conducted in a layer-wise manner in denoising networks
+ ***adaptive layer normalization (adaLN)***: follows the widespread usage of adaptive normalization layers in GANs, 
	**Scalable Diffusion Models** explores replacing standard layer norm layers in transformer-based diffusion backbones with adaptive layer normalization. Instead of directly learning dimension-wise scale and shift parameters, it regresses them from the sum of the time embedding and conditions.

**Condition Diffusion on Images**:
*SDEdit* conditions on a styled images to make image-to-image translation, while *LDM*  unifies these semantic conditions with flexible latent diffusion. 
-> Kindly note that if conditions and diffusion targets are of different modalities, pre-alignment is a practical way to strengthen the guided diffusion.