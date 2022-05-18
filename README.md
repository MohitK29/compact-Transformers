<h1>GANs based on Compact Vision Transformers</h1>

<h3> <u> OVERVIEW: </u> </h3>


![Architecture](https://user-images.githubusercontent.com/47019139/168947537-c36ee817-8490-4d84-a072-dd0d54112b9c.png)


-------------------------------------------------------------------------------------------------------- 
<h3> ABOUT THE REPO: </h3>
<h3> 1. File structure </h3>
CONFIGS – This folder contains helper codes. <br><br>
METRICS – This folder contains the code for loss functions of CVTGANs. <br><br>
MODELS – This folder contains different models which are required for training. <br><br>
MODELS_SEARCH – This folder contains CVTGANs - Generator and Discriminator models. <br><br>
PLOTS – This folder contains plots for train/test loss and accuracy. <br><br>
SRC – This folder contains necessary codes for CVT, CCT and ViTs. <br><br>
UTILS – This folder contains various miscellanous code files. <br><br>
UTILSCVT – This folder contains helper functions for CVTGANs. <br><br>

VITGAN.sbatch – sbatch file for running experiment for ViTGAN <br><br>
VitGanTrain.py – Code for training the ViTGAN. <br><br>
animation.gif – This is an example GIF of output images generated from CVTGANs Generator. <br><br>
cifar10.sbatch – sbatch file for running CCT/CVT experiment on CIFAR-10 dataset. <br><br>
cifar100.sbatch – sbatch file for running CCT/CVT experiment on CIFAR-100 dataset. <br><br>
cvtGanConfig.py – Config file for CVTGAN. <br><br>
cvtGanTrain.py – Code for training the CVTGAN. <br><br>
cvtGanTrain.sbatch – Sbatch file for training CVTGAN submitted as a SLURM job. <br><br>
cvtTrain.py – Code for training CVT for image classification. <br><br>
datasets.py – Contains initialization of datasets. <br><br>
functions.py – Helper functions for CVT GANs. <br><br>
vitgan.jpg - Sample image generated from ViTGAN. <br><br>

-------------------------------------------------------------------------------------------------------- 
<h3> 2. How to clone </h3>
(Make sure you have Git Bash installed)
Run a Git Bash terminal in the folder you want to clone in and use the following command: <br>
git clone https://github.com/MohitK29/compact-transformers.git <br>
  
-------------------------------------------------------------------------------------------------------- 
<h3> 3. How to run the code </h3>

<h4> Train the model: How to train the model using SLURM jobs on HPC </h4><br><br>
<li>Step 1: Create a conda environment <br>
Check out the following link to do so:  <br>
Managing environments — https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/packages/conda-environments <br><br>
<li>Step 2: For installing requirements
Run the following commands to set up the environment: <br><br>
pip install -r requirements.txt <br><br>
<li>Step 3: Run the following SLURM command by running the following command ‘<dataset>.sbatch’ or VITGAN.sbatch: <br>
sbatch <datset>.sbatch <br><br>
 
  ![ezgif com-gif-maker (2)]()


  
GIF of images from ViTGAN
![image](https://user-images.githubusercontent.com/47019139/168940903-b9e3328a-7d77-4042-a7e2-113d5b76c742.gif)
  
GIF of images from CVTGAN
![image]([https://user-images.githubusercontent.com/47019139/168952636-1eafaedd-f580-44c1-9da5-a27f6674ad9c.gif])

  <h2>REFERENCES:</h2>
<li>[1] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel, “Backpropagation applied to handwritten zip code recognition,” Neural computation, vol. 1, no. 4, pp. 541–551, 1989.
  
<li>[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, vol. 25, 2012.
  
<li>[3] K. He, X. Zhang, S. Ren, and J. Sun, “Identity mappings in deep residual networks,” in European conference on computer vision. Springer, 2016, pp. 630–645.
  
<li>[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.
  
<li>[5] A. Kolesnikov, A. Dosovitskiy, D. Weissenborn, G. Heigold, J. Uszkoreit, L. Beyer, M. Minderer, M. Dehghani, N. Houlsby, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” 2021.
  
<li>[6]  K. Lee, H. Chang, L. Jiang, H. Zhang, Z. Tu, and C. Liu, “Vitgan: Training gans with vision transformers,” arXiv preprint arXiv:2107.04589, 2021.

<li>[7] A. Hassani, S. Walton, N. Shah, A. Abuduweili, J. Li, and H. Shi, “Escaping the big data paradigm with compact transformers,” arXiv preprint arXiv:2104.05704, 2021.
  
<li>[8] “Compact vision transformer” https://github.com/SHI-Labs/Compact-Transformers.
 
<li>[9] TransGAN: Two Pure Transformers Can Make One Strong GAN - https://proceedings.neurips.cc/paper/2021/file/7c220a2091c26a7f5e9f1cfb099511e3-Paper.pdf https://github.com/VITA-Group/TransGAN
