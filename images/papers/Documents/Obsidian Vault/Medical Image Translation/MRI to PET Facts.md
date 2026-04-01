1. PET measures the brain's metabolism, whereas MRI measures the brain's anatomy. Our brain is always metabolically active, consuming approximately 20% of our daily energy budget and these metabolic processes are closely linked to the anatomy of the brain. Hence, there is a relation between both measurements that may be exploited. (-Inferring PET from MRI with pix2pix)

2. Simply training a deep neural network to translate MRI images to PET images is not feasible, because the L2 or L1 minimization is pixel-based which tends to favour averaged (i.e., blurry) output images. (-Inferring PET from MRI with pix2pix)

3. the evaluation of the quality of generated images is an unsolved problem. We evaluated the quality by using measures that are essentially pixel-based. There are two ways to improve the evaluation: 
	1. rely on medical experts that score the generated PET images. 
	2. use the generated images as input for a classication task.




