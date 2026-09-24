# One-Dimensional PSWF Approximation

This repository contains MATLAB examples using one-dimensional Prolate Spheroidal Wave Functions (PSWFs).

There are two main examples.

## 1. Direct approximation

The first example uses PSWFs to approximate known functions on the interval [-1,1].

Two test functions are considered:

- an exactly bandlimited function,
- an almost-bandlimited function.

The MATLAB code computes a finite PSWF expansion and reports the L2 approximation error.

Files:

- `PSWF_Two_Example_Approximation.m`
- `PSWF_Two_Example_Approximation.pdf`
- `prolatematrix.m`

## 2. Reconstruction from noisy data

The second example is an inverse problem.

A signal is passed through the time-band limiting operator and noise is added to the resulting data. The aim is to reconstruct the original signal from these noisy measurements.

Because the inverse problem becomes unstable when small PSWF eigenvalues are involved, regularization methods are used.

Files:

- `pswf_inverse_c10.m`
- `prolatematrix.m`
- `PSWF_Inverse_c10_Study.pdf`

## Requirements

- MATLAB
- `prolatematrix.m`

## Author

Hamed Baghal Ghaffari
