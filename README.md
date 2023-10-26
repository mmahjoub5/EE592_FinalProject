# EE592_FinalProject

This project aimed to compare the performance of various image restoration methods on a subset of images from the Lensless Learning paper dataset. The methods evaluated include the learned ADMM approach with hyperparameter tuning, a CNN denoiser, and a UNet deep reconstruction approach. The learned ADMM method with hyperparameter tuning outperformed the standard ADMM approach, with faster convergence and better results. However, the CNN denoiser and UNet had poor results, with overfitting and limited data. The main limitation of the ADMM method was its computational cost, as it required significant time to run. In conclusion, the learned ADMM approach with hyperparameter tuning using Total variation difference regularization showed promising results for image restoration, while the CNN denoiser and UNet approaches may require additional data and improvements to their architectures to be effective.

# Results 

## Target 
<img width="169" alt="image" src="https://github.com/mmahjoub5/EE592_FinalProject/assets/55271527/bf13384f-0eaf-46e2-8401-59ef3c143e52">

## ADMM
<img width="156" alt="image" src="https://github.com/mmahjoub5/EE592_FinalProject/assets/55271527/6438cd9a-3f75-4af1-8fa5-c15be1347bd4">


## Learned ADMM via Hyper Parameter Tuning 

<img width="158" alt="image" src="https://github.com/mmahjoub5/EE592_FinalProject/assets/55271527/abb1171c-0365-4525-8d74-8a65782ae4d3">


## Leanred ADMM via CNN Denioser 
<img width="141" alt="image" src="https://github.com/mmahjoub5/EE592_FinalProject/assets/55271527/65b6d94d-ed0c-46f8-be52-a16b72abcb8f">



# References
[1]	H. K. Aggarwal, M. P. Mani, and M. Jacob, “MoDL: Model-based deep learning architecture for inverse problems,” IEEE Trans. Med. Imag., vol. 38, pp. 394–405, 2019.
[2]	Kristina Monakhova, Joshua Yurtsever, Grace Kuo, Nick Antipa, Kyrollos Yanny, and Laura Waller, "Learned reconstructions for practical mask-based lensless imaging," Opt. Express 27, 28075-28090 (2019)

