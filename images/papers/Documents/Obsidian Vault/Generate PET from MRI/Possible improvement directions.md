- [x] 1. Close ema, use the original trained model for ddim sampling --> any theoretical supports? --> not working!!
- [ ] 2. Use self-condition
- [x] 3. only predict mean of the noise but not the variance --> is it good? --> now we are only using the pred start_x, without variance prediction
- [x] 4. Predict_x0 instead of predicting noise??? --> it works!!
- [x] 5. Put Squeeze & Excitement block after the MRI features
   --> SSE does not work!! or it could be the problem of putting backwards inside auto-cast, or because it needs pretrained model!!!
- [x] 7. Add classifier!
    -> if adding a pretrained classifier, what dataset this classifier should be pretrained on? --> pretrain on the PET dataset, then freeze it during MRI2PET training, only use as a logit generator
	    The binary CN/AD classifier (ResNet18) on PET data got overfitting a lot!!
		Use smaller scale of resnet, first train on 30 slices ones, then fine-tune on the single slice dataset
- [x] 8. fix the problem that it cannot be fp16
- [x] 9. get rid of variance learning, 
- [x] 10. try also predict PREVIOUS_X --> Not working!

===============07.17==================
future works:
- [x] Generate the difference map of generated PET and real PET
- [x] bidirectional Unets to reconstruct MRI from PET also
- [x] use a pretrained classifier to predict labels during training as well
- [ ] Cycle consistent training -> reconstruct MRI from PET again, add cycle consistent loss, so we can use it for MRI-only dataset training
- [ ] Use [==NEUROSTAT==](https://neurostat.neuro.utah.edu/) to post-process the PET and make some comparison --> but need to generate the whole 3D images (all the 2D slices)
- [x] Add tabular data in the bottleneck to help!

===============07.24==================
For classifier:
Easy to get overfitting
- Problem with MCI labels --> much less datapoints during training
- Train it on a much larger and different dataset? -> train on ADNI3, but use it on the diffusion models with ADNI2
- Freeze it during training or not
- Should I use the bottleneck features for classification (FC layers afterwards)

--> the loss L1/L2 will increase, that's reasonable because the model compensated for training to classify the correct labels
-> but then it may help to invrease the accuracy on the disease-specific parts on the synthetic images

- [ ] Check if the error maps have some characters for specific diseases
	- [ ] when evaluation, print the diagnosis labels on the corner of the generated images and the error map!
- [ ] Check if the model improves on the specific ROIs for different labels 
	- read the related papers and the feature nii
- [ ] How to use the MCI labels
- [x] Adding tabular data into the model
	- [x] Can the tabular data be added as the time embedding? -> sum its embedding with time embedding on every block
	- [x] Or only concat the tabular embedding with bottleneck
- [ ] ==How stable diffusion handle the condition and tabular input??
- [x] Bidirectional training
	- [x] What if I use the same model for both MRI->PET and PET->MRI?? Does it make sense? --> can be, but still a bit skeptical --> can be promising
	- [x] Or use 2 different generators for MRI->PET and PET->MRI, then wish the **synthetic** PET can recover the original MRI! --> might be plausible, inspired by cycle GAN
	- Or how about just try to:
		1. For a given image in domain X, apply the diffusion process to create a noisy version.
		1. Use the CycleGAN generator G to translate the noisy image into the style of domain Y.
		2. Apply the reverse diffusion process to the generated image to obtain the final translated image.
		3. Compute the cycle consistency loss by translating the final image back to domain X using the other CycleGAN generator F and applying the reverse diffusion process.
- [x]  Generate a whole 3D one --> just need to train on all slices and then generate each slice from the corresponding MRI slices
	- [x] One question: in MRI is 193,229,193, PET is 113,137,113, how can they correspond to each other on the template??
		- there should be some registration happned there that the GM is the same resolution
- [x]  What if I use grey matter density of MRI instead??
	- [x] But it seems that ADNI2/MRI does not have VBM? Should I generate it by myself? -- Yes please

test SSIM and PSNR by input two identical images

==07.31==
- [ ] Using C3D toolkit for resizing
- [x] Try to use GM map as another channel or replacement of MRI
	- [x] Use GM instead, seems not working quite well
		Based on paper *'Demystifying T1 to PET Image Translation'*, brain tissue information is the key to PET image synthesis, the performance of the U-Net-based translation model which is directly trained on tissue maps as input is competitive to the original translation model (i.e., trained by taking T1-MR images as input
- Check HALOS which also include tabular data in Unet -> no BN?
- MONAI
- [x] How about using a pretrained model as a base already (a diffusion model pretrained on ImageNet)
	- [x] -> Tried but the model OpenAI used was too large for our machine
- [x] Run comparable models as a comparison, e.g. Pix2Pix
	- [ ] Maybe use MONAI
- [x] Use L1 loss during training instead
- [x] replace the Batch Normalization with the Instance Normalization in the tasks of cross-modality synthesis (from Cross-modality Synthesis from MRI to PET Using Adversarial U-Net with Different Normalization)
	- [x] Actually we always use GroupNorm(32)
	- [x] How about change it to InstanceNorm (GroupNorm when group number = channel number)
- [ ] Try more about squeeze & excitement
- [x] Predict v instead? possibly sharper images --> not really
- [ ] Check out how to use spaced diffusion to achieve DDIM!
- [x] Try the min-SNR trick --> but does it mainly focus on ignoring small details? -> probably, since the results don't show anything better
- [ ] Different loss weighting techniques, worth comparing
- [ ] Distillation?
- [ ] Seems like we need wider unet to perform well on GM -> turn unet_dim = 128 to try -> or L1 loss makes a difference?
- [ ] need to learn the variance??
- [ ] Try a simple VAE as baseline for MRI2PET?
- [x] Downsample standardized MRI and use it as condition still
- [ ] Give diffusion autoencoder another chance? Or as a comparison
- [ ] DDIB!
- [ ] CycleGAN
- [ ] consistency model
- [ ] Slices: 60 or 30 or all the slices? --> apparently our model is not good at synthesizing more slices, so we'd better increase the model complexity!! Also check how the other papers use the slices!!
	- [ ] Cycle consistency loss helps, but it takes a lot of resources 
	- [ ] Could be because there is still non-brain parts in PET, so there needs to be a way to remove the non-brain parts in PET!! --> *Why is there no skull-stripping for PET images?? Any reason* --> ==remember to check the ADNI PET processing==https://adni.loni.usc.edu/methods/pet-analysis-method/pet-analysis/
	- [ ] To do skull-stripping on PET -> Synthstrip from freesurfer? https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip
	- [x] Try to use the MRI that also with skull now! (from '/mnt/nas/Data_Neuro/ADNI_SEG/registered_output_mni152/FS53/xx/mri.nii.gz')
	- [ ] use all slices --> *need to see if the generated whole volume is consistent, see it in slicer*
- [ ] Improve the way to add the MRI feature maps as condition: right now they are multiplied with the generation feature map, what if using a hybrid way to both multiply and somehow shift add it? 
- [ ] Maybe try Diffusion-GAN???
- [x] Better to normalize the image intendsity to [-1, 1]??
		**Important:** Images must be normalized in the range [-1, 1], as our network will have to predict noise values that are normally distributed?? --> true!

- add the information of slice position during training so it will be more consistent
	- Using attention mechanism
	- [x] using squeeze and excitement
- [x] register the MRI and PET together -> need a good registration
- [x] include the neighboring slices also as input
- need to find a shape-aware way!!

- [x] take the previous and following slices into consideration as input, then take an average of all these slices
- Take a smooth of the final PET
- use different seeds to generate several scans and take an average
- use half the sphere to make the process fast

- check exactly how to use the slice index with SE block

- [ ] Try to Use MS-SSIM loss to facilitate structural integrity at multiple scales!! (in torchmetrics)