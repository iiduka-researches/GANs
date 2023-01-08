# Existence and Estimation of Critical Batch Size for Training generative adversarial networks of two time-scale update rule
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for DCGAN, WGAN-GP, and BigGAN.

# Abstract
Previous results have shown that a two time-scale update rule (TTUR) using learning rates, such as constant and decaying learning rates, is practically useful for training generative adversarial networks (GANs) from the viewpoints of theory and practice. Moreover, not only the setting of learning rate but also the setting of batch size are important factors for training GANs of TTUR and influence the number of steps needed for training GANs of TTUR. This paper studies the relationship between batch size and the number of steps needed for training GANs of TTUR using constant learning rates. In theoretical parts, we show that, for TTUR using constant learning rates, the number of steps needed to find stationary points of loss functions of both a discriminator and a generator decreases as the batch size increases and that there exists a critical batch size minimizing the stochastic first order oracle (SFO) complexity.
In practical parts, we use the Fre ́chet inception distance (FID) as the performance measure for training GANs of TTUR, and we provide numerical results indicating that the number of steps needed to achieve a low FID score decreases as the batch size increases and that the SFO complexity increases once the batch size exceeds the measured critical batch size. Moreover, we show that measured critical batch sizes are close to estimated batch sizes based on our theoretical results.
