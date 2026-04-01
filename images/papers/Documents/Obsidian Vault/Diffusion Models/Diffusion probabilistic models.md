A diffusion probabilistic model (diffusion model) is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time

$p_\theta (x_0) = \int p_\theta (x_{0:T}) dx_{1:T}$
- reverse process: $p_\theta (x_{0:T})$
- $p_\theta (x_{0:T}) = p(x_T) \sum_{t=1}^T p_\theta (x_{t-1} | x_t)$
-- $p(x_T) = \mathcal{N} (x_T ; 0,I)$
-- $p_\theta (x_{t-1} | x_t) = \mathcal{N} (x_{t-1} ; \mu_\theta (x_t, t), \Sigma_\theta (x_t, t))$

- forward process: fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_1$, ..., $\beta_T$
![[Pasted image 20230418144349.png]]

- Training is performed by optimizing the usual variational bound on negative log likelihood:
![[Pasted image 20230418144557.png]]



Destroy all structure in data using diffusion process
data distribution -> Forward diffusion -> noise distribution

Recover structure in dataa using reversal of diffusion process
Noise distribution -> reverse diffusion -> data distribution
--> Get **learned drift and covariance functions**

**Training the reverse diffusion process with an asymptotically tight variational bound**
- model probability
- annealed importance sampling / Jarzynski equality
- Log likelihood
- Jensen's inequality
-> Get training objective
-> As variantional lower bound
*In the end turns unsupervised learning into regression problem*

**Diffusion in continuous time by stochastic differential equations (SDE)**
Forward SDE: $dx = f(x,t)dt + g(t)dw$ 
    deterministic drift + stochastic diffusion

Reverse SDE: $dx = [f(x,t) - g^2(t) ~\nabla_x log~ p_t (x)]dt + g(t) ~dw$
                                                score function

**Sampling with numerical SDE solvers:**
- approximate the reverse SDE with our score-based model:
$s_\theta (x,t) \approx \nabla_x log p_t (x)$
$dx = [f(x,t) - g^2(t) s_\theta (x,t)]dt + g(t) dw$
- numerical SDE solvers, e.g. Euler-Maruyama solver (discretezation)

**Probability flow (Ordinary) ODE**
$dx = [f(x,t) - \frac{1}{2} g^2(t) \nabla_x log~ p_t (x)]dt$

**Probability flow ODEs as continuous normalizing flow**
Efficient 

**Controllable Generation**
control signal y -> $x_0$ -> $x_t$ -> $x_T$
conditional reverse-time SDE via unconditional scores
$dx = [f(x,t) - g^2(t) \nabla_x log~ p_t (x | y)]dt + g(t) dw$
$dx = [f(x,t) - g^2(t) \nabla_x log ~p_t (x) - g^2(t) \nabla_x log ~p_t (y | x) ]dt + g(t) dw$
                unconditional score         trained separately or specified with domain knowledge




