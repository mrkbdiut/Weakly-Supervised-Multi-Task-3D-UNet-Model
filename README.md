# Multitasking-3D-UNet-PyTorch
This repository is an implementation of [ Cochlear Implant Fold Detection in Intra-operative CT Using Weakly Supervised Multi-task Deep Learning] [https://arxiv.org/abs/1606.06650](https://doi.org/10.1007/978-3-031-43996-4_24)

![image](https://github.com/mrkbdiut/Weakly-Supervised-Multi-Task-3D-UNet-Model/assets/36138901/97d6046a-9ff8-4ce8-bccd-0960960c0f12)

The model has two branches: the Segmentation Branch and the Classification Branch. In our implementation the segmentation branch segments the metal electrode array from the input 3D CT image. The classification branch attached to the bottleneck of the 3D UNet consists of several residual blocks. This branch determines whether the input 3D CT image has a tip-foldover condition.

You can find the PyTorch version of the Multi-task 3D UNet model in `model2.py`

Training and testing is done using the `train_and_test.py` file

## Environment
+ Python 3.7
+ PyTorch 1.8.1

Khan, M.M.R., Fan, Y., Dawant, B.M., Noble, J.H. (2023). Cochlear Implant Fold Detection in Intra-operative CT Using Weakly Supervised Multi-task Deep Learning. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14228. Springer, Cham.
[https://arxiv.org/abs/1606.06650](https://doi.org/10.1007/978-3-031-43996-4_24)
