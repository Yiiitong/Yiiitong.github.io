1. During training, in each epoch:
	1. A **random** time step **_t_** will be selected for each training sample (image).
	2. Apply the Gaussian noise (corresponding to **_t_**) to each image.
	3. Convert the time steps to embeddings (vectors).
	![[Pasted image 20230724082552.png|300]]
	![[Pasted image 20230724082801.png|400]]

2. During sampling:
 ![[Pasted image 20230724083316.png|500]]