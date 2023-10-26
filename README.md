# EE592_FinalProject

This project aimed to compare the performance of various image restoration methods on a subset of images from the Lensless Learning paper dataset. The methods evaluated include the learned ADMM approach with hyperparameter tuning, a CNN denoiser, and a UNet deep reconstruction approach. The learned ADMM method with hyperparameter tuning outperformed the standard ADMM approach, with faster convergence and better results. However, the CNN denoiser and UNet had poor results, with overfitting and limited data. The main limitation of the ADMM method was its computational cost, as it required significant time to run. In conclusion, the learned ADMM approach with hyperparameter tuning using Total variation difference regularization showed promising results for image restoration, while the CNN denoiser and UNet approaches may require additional data and improvements to their architectures to be effective.


References
[1]	H. K. Aggarwal, M. P. Mani, and M. Jacob, “MoDL: Model-based deep learning architecture for inverse problems,” IEEE Trans. Med. Imag., vol. 38, pp. 394–405, 2019.
[2]	Kristina Monakhova, Joshua Yurtsever, Grace Kuo, Nick Antipa, Kyrollos Yanny, and Laura Waller, "Learned reconstructions for practical mask-based lensless imaging," Opt. Express 27, 28075-28090 (2019)

