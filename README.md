# CAT-GAN
Generative  Adversarial  Network  (GAN)   is  a  class  of  unsupervised  machine  learning  systems  inwhich two different neural networks compete in a game (such as a zero-sum game) in order to train eachother.  These two are calledgeneratoranddiscriminator.  The former one has the goal to create asoutput samples with the same distribution of the training data, taking as input a vector from a randomdistribution. The latter one has to judge whether its input is a real image or a generated one, thus it willhave two classes as output (real or fake). 

In the context of Deep learning,  many variants of GAN include convolutional layers to increase thecapacity of these models for unsupervised tasks. This repository contains some of these implementa-tions.  These include DCGAN, WGAN and ProGAN.The task for these networks is to learn to generate cats’ faces, using a collection of open source data-sets. The final dataset can be found at this link: https://github.com/Ferlix/Cat-faces-dataset . 
To get moreinsights on the quality and diversity of the generated images, two different metrics are used, namely Inception Score (IS) and Fr ́echet Inception Distance (FID), and their scripts can be found in the folder "Metrics". 
Paper can be found [here](https://github.com/remosasso/Generating-cats-with-GANs/blob/master/CAT-GAN.pdf).

# Results


 ⠀ | Samples
------------ | -------------
Dataset cats (real) | ![REAL](/Results/reals_sample.png)
DeepConvolutional-GAN cats (generated)| ![dc](/Results/dcgan_result.png)
Wasserstein-GAN cats (generated) | ![WGAN](/Results/wcats.png)
Progressive-GAN (generated)| ![pro](/Results/progan_results.png)

