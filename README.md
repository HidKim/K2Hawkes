# Python Code for Kernel Method-based Triggering Kernel Estimator via Least Squares Loss
This library provides a kernel method-based triggering kernel estimator for a linear Hawkes process, implemented in Tensorflow. This method is based on a representer theorem that emerges under the principle of penalized least squares minimization. For details, see our ICLR2026 paper [1].

The code was tested on Python 3.10.8, tensorflow-deps 2.10.0, tensorflow-macos 2.10.0, and tensorflow-metal 0.6.0.

# Installation
To install latest version:
```
pip install git+https://github.com/HidKim/K2Hawkes
```

# Basic Usage
Import our tiggering kernel estimator class:
```
from HidKim_K2IE import k2_hawkes_rfm
```
Initialize RFM kernel
```
ker = kernels_rfm(n_dim, kernel='gaussian', n_rand_feature=500, seed=0, qmc=False)
```
- `n_dim`:  *int* <br>
  >The dimensionality of inputs.
- `kernel`: *string, default='gaussian'* <br> 
  >The kernel function: 'gaussian', 'laplace', and 'cauchy'.
- `n_rand_feature`:  *int, default=0* <br>
  >The number of random Fourier features.  
- `seed`:  *int, default=0* <br>
  >The seed for sampling Fourier features.
- `qmc`:  *bool, default=False* <br>
  >Quasi-Monte Carlo method is applied to RFM generation.

Import K<sup>2</sup>IE class:
```
from HidKim_K2IE import k2_intensity_estimator
```
Initialize K<sup>2</sup>IE:
```
k2ie = k2_intensity_estimator(kernel=ker)
```
- `kernel`: *kernels_rfm instance* <br> 
  
Fit K<sup>2</sup>IE with data:
```
time = model.fit(d_spk, d_region, a, b)
```
- `d_spk`: *ndarray of shape (n_points, dim_points)* <br>
  > The training point data.  
- `d_region`: *ndarray of shape (n_subregion, dim_points, 2)*  <br>
  >The observation region. e.g.) [ [[0,1],[0,1]], [[1,3],[0,1]] ] represents that there are two adjacent subdomains: one is a unit square, and the other is a rectangle with a length of 2 in the x-direction and a length of 1 in the y-direction.
- `a`: *float* <br>
  >The amplitude hyper-parameter for shift-invariant kernel function, or the regularlization hyper-parameter '\gamma' in ICML2025 paper.
- `b`:  *ndarray of shape (dim_region,)*  <br>
  >The scale hyper-parameter for shift-invariant kernel function. If a scalar value is provided, the shift-invariant kernel will be regarded as being isotropic. 
- **Return**: *float* <br>
  >The execution time.

Evaluate the integral of the squared intensity function over a specified domain (used for closs-validation of hyper-parameter):
```
int_sq = k2ie.predict_integral_squared(region)
```
- `region`: *ndarray of shape (n_subregion, dim_points, 2)* <br>
  > The region for integral.  
- **Return**: *float* <br>
  >The evaluated itengral of the squared intensity function.

Predict intensity function on specified inputs:
```
r_est = model.predict(x)
```
- `x`: *ndarray of shape (n_points, dim_points)* <br> 
  >The points on input space for evaluating intensity values.
- **Return**: *ndarray of shape (n_points,)* <br>
  >The predicted values of intensity function at the specified points.

# Reference
1. Hideaki Kim, Tomoharu Iwata. "A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization", *International Conference on Learning Representations*, 2026.
```
@inproceedings{kim2026arepre,
  title={A Representer Theorem for Hawkes Processes via Penalized Least Squares Minimization},
  author={Kim, Hideaki and Iwata, Tomoharu},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
``` 

# License
Released under "SOFTWARE LICENSE AGREEMENT FOR EVALUATION". Be sure to read it.

# Contact
Feel free to contact the author Hideaki Kim (hideaki.kin@ntt.com).