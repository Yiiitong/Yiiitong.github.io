- [x] 1. Condition on MRI, just as another channel for input to Unet -> not working, got just blurry images -> more steps (2000) also didn't help!!
- [x] 1.1. Add loss for both (pred_noise + pred_x0) -> not working
- [x] 1.2. now just put the MRI as a single channel in the last resnet block of Unet

- [x] 2. Use an encoder to encode MRI to 1-D features and then multiply it with each block of the Unet
- [x] 3. add more images -> get 60 / 30 slices per image
- [x] 4. Double Unet: one for MRI to encode feature maps on different scales, another for PET. Also add reconstruction loss for MRI to supervise its reconstructed effects
- [x] 5. Add classifier --> first pretrain a classifier for PET diagnosis? or put the classifier inside the model

Possible method: 
**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models**

Possible Baseline method:
**DUAL-GLOW: Conditional Flow-Based Generative Model for Modality Transfer (ICCV 19)**
Possible diretion: use 'Squeeze and Excitement' in Unet  --> ‘Squeeze & Excite’ Guided Few-Shot Segmentation of Volumetric Images
Possible direction: stable diffusion encoding methods

Possible improvement: *Self-conditioning* --> propose a technique for the model to directly condition on previously generated samples of its own during the iterative sampling process, which can significantly improve the sample quality of diffusion models
(https://arxiv.org/pdf/2208.04202.pdf) --> A simple implementation of Self-Conditioning is to concatenate $x_t$ with previously estimated $x˜_0$.

A.2. Is this an inverse problem???
- [x] 3. Figure out about the evaluation metrics:

def compute_reconstruction_metrics_single(target, pred):
    # target = target / target.max() + 1e-8
    # pred = pred / pred.max() + 1e-8
    # range = np.max(target) - np.min(target)
    target = target - target.min()
    pred = pred - pred.min()
    range = target.max()
    try:
        rmse_pred = skimage.metrics.mean_squared_error(target, pred)
        # rmse_pred = skimage.metrics.normalized_root_mse(target, pred)
    except:
        rmse_pred = float('nan')
    try:
        # psnr_pred = skimage.metrics.peak_signal_noise_ratio(target, pred)
        psnr_pred = skimage.metrics.peak_signal_noise_ratio(target, pred, data_range=range)
    except:
        pdb.set_trace()
        psnr_pred = float('nan')
    try:
        # ssim_pred = skimage.metrics.structural_similarity(target, pred)
        ssim_pred = skimage.metrics.structural_similarity(target, pred, data_range=range)
    except:
        ssim_pred = float('nan')
    return {'ssim': ssim_pred, 'rmse': rmse_pred, 'psnr': psnr_pred}


- [x] 4. How about using more data? Now there are 1300 data for 3D, if we use all 3 angles we can have 3900 data
- [ ] Need to learn more abouot classifier guidance and CLIP model