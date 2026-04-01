**1. GAN‑based synthetic brain PET image generation**, *Jyoti Islam and Yanqing Zhang*
+ proposed model can create brain PET images for three different stages of AD — CN, MCI, and AD.

**2. Synthetic PET via Domain Translation of 3-D MRI**
+ we use a dataset of 56 F-FDG-PET/MRI exams to train a 3-D residual UNet to predict physiologic PET uptake from whole-body T1-weighted MRI
![[Pasted image 20230509190034.png]]

**3. Bidirectional Mapping Generative Adversarial Networks for Brain MR to PET Synthesis**
![[Pasted image 20230509190843.png]]
reimplement the idea in diffusion models?
+ In order to guarantee the variability of the generated results, the latent vector sampled from a standard normal distribution is provided along with the MR image as an additional input of the generator.
+ it provides a connection between the brain PET images and the latent vectors, which encourages the generator to synthesize the perceptually realistic PET images while preserving the diverse details of brain structures in different subjects.
+ This mechanism explicitly encourages a bidirectional consistency between the latent space and the PET images, so that the semantic information of PET images is embedded into the high-dimensional latent space.
+ the forward mapping starts with encoding the PET images into the latent space. To ensure the encoded latent vector has a similar distribution with the sampled latent vector, the encoded vector is trained to conform to the standard normal distribution by optimizing the KL divergence. 
+ The generator then tries to map the input MR and the encoded vector to the synthetic PET images. 
+ In the backward mapping, the generator is first used to synthesize PET images from the MR and sampled latent vectors. Subsequently, the synthetic PET image is fed to the encoder to reconstruct the input latent vector.

**4. Inferring PET from MRI with pix2pix**
+ using paired MRI and PET brain scans of Alzheimer's disease patients and healthy controls (from *ADNI*)
+ using data slices, generate synthetic PET slices