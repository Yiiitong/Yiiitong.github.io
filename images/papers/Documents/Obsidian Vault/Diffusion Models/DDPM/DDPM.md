**During training:**
1. Get image data from the dataloader (no noise)
2. Select random timestep (different for each sample)
3. Generate random noise (torch.randn())
4. Generate noisy image on step t (forward process)
5. Calculating MSE between the actual moise and the predicted noise (gradient descent step) 

**During inference:**
1. sampling random noise as $x_T$
2. reverse ordinal sampling t from T to 1
3. sampling a random noise as base noise
4. model's prediction of the added noise
5. calculating the previous sample

**Conditioning:**
1. adding a separate channel
2. concatenating the conditioning vector to the timestamp embedding
3. using another model to guide the generation process