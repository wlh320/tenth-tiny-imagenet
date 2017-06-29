# tenth-tiny-imagenet

solve one-tenth of Tiny Imagenet Challenge in python

## Dataset

one-tenth of [Tiny Imagenet](https://tiny-imagenet.herokuapp.com/)

I randomly choose 20 classes of this dataset, with 500 train images and 50 validation images of each class.

Exactly one-tenth of this dataset.

## Experiment 1: HSV Histogram + HOG

I extract the color features and HOG features from images, then train with a simple MLPClassifier.

Result:

212 seconds spent for training, get accuracy 45.3% on validation set.


## Experiment 2: CNN

The network is almost the same as the example given by Keras for solving CIFAR-10.

Result:

20 epochs, 1194 seconds spent for training, get accuracy 53.3% on validation set.


## Conclusion

The model is very simple, and the result is not good. But I learned a lot from it.

If I have more spare time or a more powerful laptop in the future, I will improve it. XD
