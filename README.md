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
- `spk`:  *ndarray of shape (n_points, 2)* <br>
  > The training time-point data: [time, dimension].\
  e.g.) [ [0.1, 0], [0.2, 2], [0.6, 1] ] represents that events occurred at times 0.1, 0.2, and 0.6 in the 1st, the 3rd, and the 2nd dimensions of a Hawkes process, respectively.
- `T`: *float*  <br>
  >The end of observation region [0, T].
- `gamma`: *float* <br>
  >The regularlization hyper-parameter '\gamma' in ICLR2026 paper.
- `b`:  *float*  <br>
  >The scale hyper-parameter for shift-invariant kernel function.\
  e.g.) 'gaussian' kernel: k(t,t') = exp[-(b(t-t'))^2]. 
- `support`:  *float*  <br>
  >The support window for the triggering kernels. 
- **Return**: *float* <br>
  >The execution time.

Predict triggering kernel on specified inputs:
```
trig_est = k2h.predict(x, edge)
```
- `x`: *ndarray of shape (n_points,)* <br> 
  >The points on input space for evaluating triggering kernel values.
- `edge`: int, *ndarray of shape (2,)* <br> 
  >The pair of dimensions that specifies the interaction direction. The 1st dimension is '0'. \
  e.g.) [0, 2] represents the triggering kernel from the third dim to the 1st dim. 
- **Return**: *ndarray of shape (n_points,)* <br>
  >The predicted values of the specified triggering kernel at the specified points.

Predict baseline intensities:
```
mu_est = k2h.get_mu()
```
- **Return**: *ndarray of shape (n_dimensions,)* <br>
  >The predicted values of baseline intensities.

Evaluate intensity functions on specified inputs under the estimated triggering kernels:
```
r_est = k2h.intensity(x, spk)
```
- `x`: *ndarray of shape (n_points,)* <br> 
  >The time-points for evaluating intensity values.
- `spk`: *ndarray of shape (n_points, 2)* <br>
  >The event data observed during the period of interest for intensity estimation.
- **Return**: *list of ndarray of shape (n_dimensions, (n_points,))* <br>
  >The predicted values of intensity functions at the specified points.

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