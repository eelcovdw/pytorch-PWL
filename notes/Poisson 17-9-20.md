# Aim
Make a relaxed poisson PWL, with properties:

- Something like Spline(round(k) | \lambda) == Poisson(round(k) | \lambda)
- Not possible with simple lines between each PMF point: not enough DOF
- New idea: Extend current BPWL shape to Poisson, fit 2 half BPWLs around each integer k, which integrates to Poisson(k|\lambda)
- Upsides:
    - spline is continuous, since low points of spline are fixed (h_d)
    - Works the same as BPWL: Bernoulli(1 | p) = \int_0.5^1 BPWL(x | p) dx
    - Most mass is concentrated around integers
    - Relatively easy to construct
    - Really possible with any PMF that is differentiable wrt parameters
    
# Implementation notes
- First prototype is rectified Poisson with support [0, N]. Possibly extended with unrectified + exponential tail.
- Difference with BPWL: each integer has 2-sided relaxation, so support of spline is [-0.5, N+0.5]
- maybe an issue: if h_d > Poisson(k | \lambda), spline goes up instead of down at k+-0.5
    but: maybe not large issue, since h_d can be set to be tiny (1e-6) so it only happens at the tail.

# Wilker notes (slack)
If we concentrate on distributions as this "relaxed Poisson" (and think of the BPWL as special case), here are some properties that can motivate the approach:
- Ordinal outcomes
- Multimodal
- Differentably reparameterisable
- Known cdf (useful for example for giving it full pmf treatment for discrete observations)
- Known KL

#### Some applications

- Think of a VAE as a continuous mixture (or a compound distribution): \int p(z)p(x|z) dz. We can offer a highly multimodal p(z). Moreover \int p(z)p(x|z) dz might have close to \sum_{i=1}^K w_k p(x|z=k). That is, it looks more like a mixture (which is difficult to reparameterise). Perhaps we can test that by comparing the proper finite mixture and the relaxed mixture (for example in terms of KL(finite||relaxed) or some non-parametric test based on samples in data space). Of course we are talking about scalar z here (or at most a vector of independent relaxed draws).
- An alternative parameterisation of [Hoogeboom et al (2019)](https://papers.nips.cc/paper/9383-integer-discrete-flows-and-lossless-compression.pdf) integer flows. Note that here we will use STE on top of our sample (showing that we are not competing with STE).
- Semi-supervised learning (used both as likelihood and as approximate posterior): latent number of objects in some toy task? the identity of a digit in MNIST (thinking about this 'relaxation of finite mixture' point of view)?
