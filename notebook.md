# Summary for work 

## data 
+ The label is stored in an csv file, The mask is *.png
+ There is no overlapping between melanoma classification and melanoma segmentation 
+ There are ~2,000 with/out masks/labels. There are ~10,000 iamges with/out labels/masks. 

## Inputs preprocessing & data augmentation
+ network training convergence faster as the input are whited. linearly transformed to have zero means and unit variances, and decorrelated.

+ [Histogram Equalization](https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf) this method is used for adjusting image intensities to enhance contrast. 
  
    > Let ƒ be a given image represented as a $m_r$ by $m_c$ matrix of integer pixel intensities, ranging from 0 to $L-1$. L is the number of possible intensity values(256 in most case). Let $p$ denote the normalized histogram of ƒ with a bin for each possible intensity, So:
    >$$p_n = \frac{\sf number\ of\ pixels\ with\ intensity\ n }{\sf total\ number\ of\ pixels }\quad  \sf n = 0,1,...,L-1$$
    > The histogram equalized image g will be defined by:... [detail](https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf)

+ `random rotation`,`random flip`,`random brightness` 

## Related Work 
[C-Unet](https://www.biorxiv.org/content/biorxiv/early/2018/08/01/382549.full.pdf) published in Aug_1 2018 

[Sub-band Decomposition](https://arxiv.org/ftp/arxiv/papers/1703/1703.08595.pdf) this is about the Laplacian image in DNN

tutorial for [transpose convolution](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0)

For shortcut [Dense Layer](https://arxiv.org/pdf/1608.06993.pdf) and [Resnet](www.google.com)

## Graph detail
- The conv_transpose will expend height and width of it's input two times, with may cause unequal with the previous correspond layer. There will be crop and concat happened between down_path conv and up_path conv and the crop is applied on up_path. 

## Parameters Setting
params.cost_type{"dice_coefficient", "cross_entropy"}
params.optimizer{"momentum", "adam"}

>Dice_coefficient: Using soft-dice + binary_cross_entropy as loss function
---

## **TODO**

- [ ] Finish Unet Structure in 10-14-2018
- [ ] after work done , check random process to really understand [batch normalization](https://arxiv.org/pdf/1502.03167.pdf)
- [ ] 