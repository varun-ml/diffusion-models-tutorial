# Work on Density estimation using Diffusion models (Pytorch + Jax)

#### Will be demonstrating critical concepts of diffusion model using some toy 2d distribution firstly, followed by using same concepts on EMNIST datasets

*Done*

* Ground truth data estimation/ Error estimation/ Score estimation of diffusion process
* Cosine and linear schedules
* Clipping to improve stabalization of the generation process
* Time embedding to encode timestep in model
* Classifier free guidance and semi supervised model training
* Faster sampling in generation process using striding steps in denoising

*Pending*
* Learning the schedule
* EMNIST data generation using U-nets and JAX 

## Generation toy distributions using diffusion models
1. Parabola 

![alt text](gifs/para.gif "parabola generated using error estimation in denoising process")

2. Circles

![alt text](gifs/circles.gif "circles generated using error estimation in denoising process")

3. Half Moon

![alt text](gifs/moons.gif "moons generated using error estimation in denoising process")

5. Circles + half-moon

![alt text](gifs/complex.gif "Circles + Half moons generated using error estimation in denoising process")

7. Circles + moon using Clipping 

![alt text](gifs/complex_clipping_energy.gif "Circles + Half moons generated using score estimation in denoising process + using clipping")

8. Generating class-conditioned distributions

![alt text](gifs/class_conditioned_moon_circles.gif "Circles + Half moons generated using class conditioned score estimation in denoising process + using clipping")

9. Generating class-conditioned distributions (few shots only using 2k samples)

![alt text](gifs/class_conditioned_mixed_few_shot.gif "(few shot learning) Circles + Half moons generated using class conditioned score estimation in denoising process + using clipping")
