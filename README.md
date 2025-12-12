# What Happens Next? Anticipating Future Motion by Generating Point Trajectories

<p align="center">
  <a href="https://gabrijel-boduljak.com/" target="_blank" rel="noopener noreferrer">Gabrijel Boduljak</a> |
  <a href="https://karazijal.github.io/" target="_blank" rel="noopener noreferrer">Laurynas Karazija</a> |
  <a href="https://eng.ox.ac.uk/people/iro-laina" target="_blank" rel="noopener noreferrer">Iro Laina</a> |
  <a href="https://chrirupp.github.io/" target="_blank" rel="noopener noreferrer">Christian Rupprecht</a> |
  <a href="https://www.robots.ox.ac.uk/~vedaldi/" target="_blank" rel="noopener noreferrer">Andrea Vedaldi</a>
</p>

<div align="center">
  <span><a href="https://www.robots.ox.ac.uk/~vgg/">VGG, University of Oxford</a> </span>
</div>

<div align="center" >
  <strong>Abstract</strong>
  <p align="left">
    We consider the problem of forecasting motion from a single image, i.e., predicting
    how objects in the world are likely to move, without the ability to observe other
    parameters such as the object velocities or the forces applied to them. We formulate
    this task as conditional generation of dense trajectory grids with a model that
    closely follows the architecture of modern video generators but outputs motion
    trajectories instead of pixels. This approach captures scene-wide dynamics and
    uncertainty, yielding more accurate and diverse predictions than prior regressors
    and generators. Although recent state-of-the-art video generators are often regarded
    as world models, we show that they struggle with forecasting motion from a single
    image, even in simple physical scenarios such as falling blocks or mechanical
    object interactions, despite fine-tuning on such data. We show that this limitation
    arises from the overhead of generating pixels rather than directly modeling motion
  </p>
</div>

## Method

<div align="center">
  <img src="./teaser.svg" style="vertical-align: middle; margin-right: 8px; margin-bottom: 8px; " />
</div>

**An overview of our method.** Given an input image $`\mathbf{I}`$ and a grid of query points, we predict $`T`$ future point trajectories. Rather than generating raw point trajectories, for computational efficiency, we operate within the latent space of a trajectory VAE (encoder $`\phi`$, decoder $`\psi`$). Specifically, we employ a latent flow matching denoiser to generate trajectory latents $`\mathbf{z} \in \mathbb{R}^{T \times h \times w \times d}`$. conditioned on the input image $`\mathbf{I}`$ DINO patch features $`\mathbf{f}`$. These are subsequently decoded into a final grid of point trajectories $`\mathbf{x} \in \mathbb{R}^{T \times H \times W \times 2}`$. Our method generates diverse and plausible future motion.

## Instructions

**Will be released soon.**
