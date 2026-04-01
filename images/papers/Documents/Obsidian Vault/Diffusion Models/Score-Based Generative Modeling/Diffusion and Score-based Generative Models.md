(https://www.youtube.com/watch?v=wMmqCMwuM2Q)
Generative models -> model distrbutions close to the data distribution
Challenge: *tackling the intractable normalizing constant*
Previous Methods:
- apporximating the normalizing constant (energy-based models)
- using restricted NN models (normalizing flow models, VAE)
- modeling the generation process only (GAN)

New proposal: **working with score functions**
probability density function: $p(x)$
(Stein) score function: $\nabla_x log p (x)$ -> derivative of distribution 

$\nabla_x log ~p (x) = \nabla_x f_\theta (x) - \nabla_x log ~Z_\theta$
$= \nabla_x f_\theta (x)$
$= s_\theta (x)$ -> score model
-> score functions bypass the normalizing constant

Score models can be estimated from data:
Given ${x_1, ...x_N}$ ~ $p_{data}(x)$
Goal: $\nabla_x log p_{data} (x)$
Score model $s_\theta (x) \approx \nabla_x log p_{data} (x)$
-> train the score model so that it's close to the data distribution
How to train = how to compare two vector fields of scores
-> Fisher divergence
![[Pasted image 20230418152500.png|300]]
However we don't know the GT of $\nabla_x log p_{data} (x)$
-> Score Matching to solve the problem, above is equivalent to:
![[Pasted image 20230418154223.png|300]]

Another solution: **Denoising score matching**

Then to generate new data point -> sampling from score functions
Given the score fundtion, if we just follow the scores, it will just converge to one point
So instead we need to follow the noisy scores 
-> **Langevin dynamics**:
- inject Gaussian noise to the score function and follow the noise perturbed score functions
- if we keep the sampling procedure long enough and keep the step very small, then Langevin dynamics will gurantee to give us correct samples from the score function
-> Langevin dynamics can sample data points from a probability density distribution using only the score ∇log⁡(q(x)) in an iterative process.

Langevin dynamics sampling:
- sample from p(x) using only the score $\nabla_x log p (x)$
- initialize $x^0$ ~ $\pi(x)$ (simple distribution)
- repeat for t <- 1,2,...,T, $z^t$ ~ $\mathcal{N}(0, I)$
- ![[Pasted image 20230418160448.png|300]]
- ![[Pasted image 20230418160540.png|400]]
- $s_\theta (x) \approx \nabla_x log ~p (x)$
-> score matching + Langevin dynamics
*-> However it did not work!*
Because:
**Langevin dynamics will have trouble exploring low data density regions (snice score model just compare difference between original data and newly generated ones)**

One solution: improving score estimation by adding noise (*perturbed density*)
-> high noise provides useful directional info for langevin dynamics
-> but perturbed density no longer approximates the true data density

Solution: using multiple noise levels
**Noise conditional score model**: -> give weights for different noise level models
![[Pasted image 20230418165103.png|300]]
-> a generalization to the training objective of diffusion probabilistic models

Sampling proceudre: *annealed Langevin dynamics* (sampling from the one with largest noise level until the one with least noises)

## Control the generation process
Borrow from Bayes' rule for score functions:
Bayes' rule:
$p(x|y) = \frac{p(x)~p(y|x)}{p(y)}$
Bayes' rule for score functions: (condition: y, can be labels, masks...)
![[Pasted image 20230418170113.png|300]]
-> we can plug in different forward models for the same score model
-> just need to train the unconditional score model once, then repurpose it for various conditional generation applications, just by switching the forward model ($p(y | x)$)

### Perturbing data with stochastic processes
Stochastic process: a collection of infinite number of random variables
To choose the stochastic process that represents an infinite number of noisy data densities -> **Stochastic differential equation (SDE)**:
$dx_t = f(x_t,t)dt + g(t)dw_t$  ($dw_t$: infinitestimal noise)
    deterministic drift + stochastic diffusion

SDE -> stochastice process -> probability densities
Forward SDE: ![[Pasted image 20230418175954.png|150]]

### Data generation via reverse stochastic processes
reverse SDE: ![[Pasted image 20230418180041.png|400]]
                                      score function + infinitestimal noise in the reverse time direction
-> then use any numerical solver to solve this reverse-time SDE for sample generation     
-> e.g., Euler-Maruyama (analogous to Euler for ODEs)
![[Pasted image 20230418181537.png|500]]


**Score-based generative modeling via SDEs:**
training objective: ![[Pasted image 20230418180559.png|400]]

To compute the accurate probability values -> *Converting the SDE to an ODE (probability flow ODE)*
ODE can also convert Gaussian distribution to real data distribution
SDE: ![[Pasted image 20230418175954.png|150]]
ODE: ![[Pasted image 20230418181203.png|200]] -> only rely on score function

-> we can solve the ODE in various ways

### Compute the exact likelihood with ODEs
![[Pasted image 20230418181943.png|500]]
-> probability flow ODE allows exact likelihood computation

*Theorem: Connection bettwen KL divergence and score matching:*
![[Pasted image 20230418182237.png|500]]
-> Efficient loss for maximum likelihood training

Probability flow ODE -> **latent space manipulation**
-> uniquely identifiable encoding -> different model architectures and weights will map the same data point to the same latent code, because:
![[Pasted image 20230418182702.png|300]]
















