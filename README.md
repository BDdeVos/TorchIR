# TorchIR: Pytorch Image Registration 
TorchIR is a image registration library for **deep learning image registration (DLIR)**. I have 
integrated several ideas for image registration.

The current version lacks a document, but I have included quite a descriptive tutorial using MNIST 
data as an example. The example experiments are light-weight and should run on any CPU, although 
running it on a GPU will increase the speed. The notebook contains a tutorial using MNIST data as 
an example. Although the code runs faster on a GPU, this tutorial is small enough to run on CPU.

For the tutorial I rely on PyTorch Lightning, which can be installed via:
> pip install pytorch-lightning

The pytorch-lightning trainer modules automatically create tensorboard log files. I store them in 
the `./output/lightning_logs` directory. Simply inspect them using:
> tensorboard --logdir=./output/lightning_logs

If you use this code for your publications, don't forget to cite my work ;)

[1] Bob D. de Vos, Floris F. Berendsen, Max A. Viergever, Marius Staring and Ivana Išgum, 
"End-to-end unsupervised deformable image registration with a convolutional neural network," 
in Deep learning in medical image analysis and multimodal learning for clinical decision support. 
Springer, Cham, 2017. p. 204-212, doi: 10.1007/978-3-319-67558-9_24
https://link.springer.com/chapter/10.1007%2F978-3-319-67558-9_24

[2] Bob D. de Vos, Floris F. Berendsen, Max A. Viergever, Hessam Sokooti, Marius Staring and Ivana Išgum
"A deep learning framework for unsupervised affine and deformable image registration," Medical image analysis, vol. 52, pp. 128-143, Feb. 2019, doi: 10.1016/j.media.2018.11.010
https://www.sciencedirect.com/science/article/pii/S1361841518300495

Please note that the code is still under heavy development and I'd really love your input.