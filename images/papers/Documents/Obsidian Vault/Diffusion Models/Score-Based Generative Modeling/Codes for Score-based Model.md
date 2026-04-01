## Time-Dependent Score-Based Model
There are no restrictions on the network architecture of time-dependent score-based models, except that their output should have the same dimensionality as the input, and they should be conditioned on time.

Several useful tips on architecture choice:

-   It usually performs well to use the [U-net](https://arxiv.org/abs/1505.04597) architecture as the backbone of the score network $s_θ(x,t)$,
    
-   We can incorporate the time information via [Gaussian random features](https://arxiv.org/abs/2006.10739). Specifically, we first sample $ω∼\mathcal{N}(0,s^2I)$ which is subsequently fixed for the model (i.e., not learnable). For a time step $t$, the corresponding Gaussian random feature is defined as
    $[sin(2πωt);cos(2πωt)]$
    where $[a⃗ ;b⃗ ]$ denotes the concatenation of vector a⃗  and b⃗ . This *Gaussian random feature can be used as an encoding for time step $t$* so that the score network can condition on $t$ by incorporating this encoding.
    
```
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
```

-   We can rescale the output of the U-net by 
	$1/E[∥∇xlogp0t(x(t)∣x(0))∥_2^2]$. 
	This is because the optimal $s_θ(x(t),t)$ has an $ℓ_2$-norm close to $E[∥∇_xlog~p_{0t}(x(t)∣x(0))]∥_2$, and the rescaling helps capture the norm of the true score. Recall that the **training objective** contains sums of the form:
    
    $E_{x(t)∼p_{0t}(x(t)∣x(0))}[∥s_θ(x(t),t)−∇_{x(t)}log~p_{0t}(x(t)∣x(0))∥_2^2]$.
    
    Therefore, it is natural to expect that the optimal score model $s_θ(x(t),t)\approx ∇_{x(t)}log~p_{0t}(x(t)∣x(0))$.
    
-   Use [exponential moving average](https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3) (EMA) of weights when sampling. This can greatly improve sample quality, but requires slightly longer training time, and requires more work in implementation.

## Training with Weighted Sum of Denoising Score Matching Objectives
+ First of all, we need to specify an SDE that perturbs the data distribution $p_0$ to a prior distribution $p_T$. We choose the following SDE:
		$dx=σ^tdw,t∈[0,1]$
In this case,
		$p_{0t}(x(t)∣x(0))=\mathcal{N}(x(t);x(0),\frac{1}{2logσ}(σ^{2t}−1)I)$
and we can choose the weighting function:   
		$λ(t)=\frac{1}{2logσ}(σ^{2t}−1)$
When $σ$ is large, the prior distribution, $p_{t=1}$ is
	$∫p_0(y)\mathcal{N}(x;y,\frac{1}{2logσ}(σ^{2}−1)I)dy≈N(x;0,\frac{1}{2logσ}(σ^{2}−1)I))$
which is approximately independent of the data distribution and is easy to sample from.

***Intuitively, this SDE captures a continuum of Gaussian perturbations with variance function $\frac{1}{2logσ}(σ^{2t}−1)$. This continuum of perturbations allows us to gradually transfer samples from a data distribution $p_0$ to a simple Gaussian distribution $p_1$.***

![[Pasted image 20230425144335.png]]

![[Pasted image 20230425144538.png]]

**Training**:

![[Pasted image 20230425145049.png]]

## Sampling with Numerical SDE Solvers
![[Pasted image 20230425145213.png]]

**Define the Euler-Maruyama sampler:**
![[Pasted image 20230425145507.png]]

## Sampling with Predictor-Corrector Methods
Aside from generic numerical SDE solvers, we can leverage special properties of our reverse-time SDE for better solutions. Since we have an estimate of the score of $p_t(x(t))$ via the score-based model, i.e., $s_θ(x,t)≈∇_{x(t)}log~p_t(x(t))$, we can leverage *score-based MCMC approaches*, such as *Langevin MCMC*, to correct the solution obtained by numerical SDE solvers.

Score-based MCMC approaches can produce samples from a distribution $p(x)$ once its score $∇_xlog~p(x)$ is known. For example, Langevin MCMC operates by running the following iteration rule for $i=1,2,⋯,N$:

$x_{i+1}=x_i+ϵ∇_xlog~p(x_i)+ \sqrt{2ϵ}z_i$,

where $zi∼\mathcal{N}(0,I)$, $ϵ>0$ is the step size, and $x_1$ is initialized from any prior distribution $π(x_1)$. When $N→∞$ and $ϵ→0$, the final value $x_{N+1}$ becomes a sample from $p(x)$ under some regularity conditions. Therefore, given $s_θ(x,t)≈∇_{x(t)}log~p_t(x(t))$, we can get an approximate sample from $p_t(x)$ by running several steps of Langevin MCMC, replacing $∇_{x(t)}log~p_t(x(t))$$ with $s_θ(x,t)$ in the iteration rule.

**Predictor-Corrector samplers combine both numerical solvers for the reverse-time SDE and the Langevin MCMC approach.** In particular, we first apply one step of numerical SDE solver to obtain $x_{t−Δt}$ from $x_t$, which is called the "predictor" step. Next, we apply several steps of Langevin MCMC to refine $x_t$, such that $x_t$ becomes a more accurate sample from $p_{t−Δt}$. This is the "corrector" step as the *MCMC helps reduce the error of the numerical SDE solver*.

**Predictor-Corrector sampler:**

![[Pasted image 20230425150549.png]]

## Sampling with Numerical ODE Solvers
For any SDE of the form $dx = f(x,t)dt + g(t)dw$, 
there exists an associated ordinary differential equation (ODE):
$dx=[f(x,t)−1/2g(t)^2∇_xlog~p_t(x)]dt$,

such that their trajectories have the same mariginal probability density $p_t(x)$. Therefore, by solving this ODE in the reverse time direction, we can sample from the same distribution as solving the reverse-time SDE. We call this ODE the _probability flow ODE_.

Below is a schematic figure showing how trajectories from this probability flow ODE differ from SDE trajectories, while still sampling from the same distribution:
![[Pasted image 20230425152324.png]]

Therefore, we can start from a sample from $p_T$, integrate the ODE in the reverse time direction, and then get a sample from $p_0$. In particular, for the SDE in our running example, we can integrate the following ODE from $t=T$ to $0$ for sample generation:
$dx=−1/2~σ^{2t}s_θ(x,t)dt$
This can be done using many black-box ODE solvers provided by packages such as `scipy`.
**ODE sampler:**
![[Pasted image 20230425152559.png]]
![[Pasted image 20230425152625.png]]

### Sampling:
![[Pasted image 20230425152821.png]]

