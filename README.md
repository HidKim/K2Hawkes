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
Initialize our estimator:
```
k2h = k2_hawkes_rfm(kernel='gaussian', n_rand_feature=200, seed=0)
```
- `kernel`: *string, default='gaussian'* <br> 
  >The kernel function: 'gaussian', 'laplace', and 'cauchy'.
- `n_rand_feature`:  *int, default=200* <br>
  >The number of random Fourier features. Quasi-Monte Carlo method is applied to random Fourier feature generation.
- `seed`:  *int, default=0* <br>
  >The seed for sampling Fourier features.
  
Fit our estimator with data:
```
time = k2h.fit(spk, T, gamma, b, support)
```
- `spk`: *list of shape (dim_processes,)* <br>
  > The training time-point data. e.g.) [ [0.2, 0.3], [0.6], [0.1, 0.5, 0.7] ] represents that two, one, and three events occurred in the 1st, the 2nd, and the 3rd dimensions of a Hawkes process, respectively.  
- `T`: *float*  <br>
  >The end of observation region [0, T].
- `gamma`: *float* <br>
  >The regularlization hyper-parameter '\gamma' in ICLR2026 paper.
- `b`:  *float*  <br>
  >The scale hyper-parameter for shift-invariant kernel function. e.g.) 'gaussian' kernel: k(t,t') = exp[-(b(t-t'))^2]. 
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