**1. Dual Diffusion Implicit Bridges for Image-to-Image Translation (ICLR 2023)** *(Xuan Su1 Jiaming Song2 Chenlin Meng1 Stefano Ermon1,3)*
	Image translation with DDIBs relies on two diffusion models trained independently on each domain, and is a two-step process: 
	  DDIBs first obtain latent encodings for source images with the source diffusion model, and then decode such encodings using the target model to construct target images. 
	Both steps are defined via **ordinary differential equations (ODEs)**, thus the process is cycle consistent only up to discretization errors of the ODE solvers. Theoretically, we interpret DDIBs as concatenation of *source to latent, and latent to target Schrödinger Bridges*, a form of entropy-regularized optimal transport, to explain the efficacy of the method.
	![[Pasted image 20230511171034.png]]
	Given a prior reference measure W1, the well-known Schr¨odinger Bridge Problem (SBP) seeks the most probable evolution across time t between the marginals p0 and p1

**2. UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion Probabilistic Models**
	+ trains a generative model to infer the joint distribution of images over both domains as a Markov chain by minimising a denoising score matching objective conditioned on the other domain
	+ we update both domain translation models simultaneously, and we generate target domain images by a denoising Markov Chain Monte Carlo approach that is conditioned on the input source domain images, based on Langevin dynamics.