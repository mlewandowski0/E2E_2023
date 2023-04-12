# End-to-End particle classification

## Task 1
The task was a binary classification. A 500000 of 32x32x2 images were provided in hdf5. Data contained equal split between two classes (Electorn and Photon). These images consisted of 2 channels : pixelwise energy of the hit and pixelwise time of the hit. Main problem with data is it's sparsity : only ~1% of the energy channel is non-zero, and ~5-10% of the hit channel is non-zero. Visualization of few randomly selected images : 
![images/task1.png](images/task1.png)
Overall goal was to achieve AUC ROC >= 80%. Models has been implemented in All models are contained in [keras](E2E_Task1_keras_final.ipynb) and [pytorch](E2E_Task1_pytorch_final.ipynb). Dataset has been splitted into two parts : 90% for training and 10% for validaton (and model performing best on validation was saved)

| name of the model  |  best AUC ROC on validaton score  |
|---|---|
| basic MLP  | 0.765  |
| CNN 1 : (LeNet based, kernel size = 3)  |  0.797 |
| CNN 2 : (LeNet based, kernel size = 5)  |  0.802 |
| CNN 5 : (convolutions with kernel sizes = [1,2,2,3,3,5,5])  |  0.81 |
| CNN 7 : (convolutions with kernel sizes = [1,2,2,3,3,5,5],larger head)  |  0.816 |
| CNN 7 : (convolutions with kernel sizes = [1,2,2,3,3,5,5],larger head)  |  0.816 |
| CNN 8 : (convolutions with kernel sizes = [1,2,2,3,3,5,5],prelu)  |  0.813 |
| CNN 8 : (convolutions with kernel sizes = [1,2,2,3,3,5,5],prelu)  |  0.813 |
|Resnet18 (from scratch,    11.173 mln parameters)|0.801|
|ConvNeXt (467k parameters)|0.802|
|Ensemble model (7 models, each model was CNN5 trained on seperate subset of data)|0.82|
|MobileViT2|0.809|
|ViT (patch merger)|0.79|

## Task 2

# References 
- https://paperswithcode.com/sota/image-classification-on-cifar-10
- https://paperswithcode.com/sota/image-classification-on-mnist

# CNN references 
- https://arxiv.org/pdf/2201.03545.pdf
- 