# Project Title

A Deep Learning model that recognizes people from a set of images. Model trained on dataset that contains around 5k images. Dataset used for this project is class imbalanced, model accuracy can be increased by using a balanced dataset.

## Getting Started
```
Repository contains:
  1-pre_processing.ipynb: Preprocess the dataset for further training
  2-generate_model.ipynb: Used to generate model
  3-prediction.ipynb: Prediction script for recognizing people
  base_model_vgg.py: Model script(Here I used Mini VGGNet)
```
### Prerequisites
```
* Activate virtual env(python3)

* Run 1-1-pre_processing.ipynb for preprocess images
	This will create train and test meta data files for further processing
	This will also create 'class_labels.csv', that contains target labels which we use later while prediction
* RUN 2-generate_model.ipynb
	This will generate model named 'model.h5' in same directory
	Trains on MiniVGGNet(base_model.py)
	User can tune the hyper parameters (epochs, batch size, optimizer, weight decay, etc..)
* RUN 3-prediction.ipynb
	This will takke random images from dataset directory for prediction
	model will predict person name
	dlib library used for detecting face
```
### Model Summary
```
```_________________________________________________________________
Layer (type)                 Output Shape              Param # 
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 32)        896
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 32)        0 
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 64, 32)        128
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 32)        9248
_________________________________________________________________
activation_2 (Activation)    (None, 64, 64, 32)        0 
_________________________________________________________________
batch_normalization_2 (Batch (None, 64, 64, 32)        128 
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0 
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 32, 32)        0 
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 64)        18496 
_________________________________________________________________
activation_3 (Activation)    (None, 32, 32, 64)        0 
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 32, 64)        256 
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 32, 32, 64)        36928 
_________________________________________________________________
activation_4 (Activation)    (None, 32, 32, 64)        0 
_________________________________________________________________
batch_normalization_4 (Batch (None, 32, 32, 64)        256 
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 16, 16, 64)        0 
_________________________________________________________________
dropout_2 (Dropout)          (None, 16, 16, 64)        0 
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 16, 16, 128)       73856
_________________________________________________________________
activation_5 (Activation)    (None, 16, 16, 128)       0
_________________________________________________________________
batch_normalization_5 (Batch (None, 16, 16, 128)       512 
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 16, 16, 128)       147584
_________________________________________________________________
activation_6 (Activation)    (None, 16, 16, 128)       0 
_________________________________________________________________
batch_normalization_6 (Batch (None, 16, 16, 128)       512 
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 8, 8, 128)         0 
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 128)         0 
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0 
_________________________________________________________________
dense_1 (Dense)              (None, 512)               4194816 
_________________________________________________________________
activation_7 (Activation)    (None, 512)               0 
_________________________________________________________________
batch_normalization_7 (Batch (None, 512)               2048
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 4676)              2398788
_________________________________________________________________
activation_8 (Activation)    (None, 4676)              0
=================================================================
Total params: 6,884,452
Trainable params: 6,882,532
Non-trainable params: 1,920
```
## References

* [Keras](https://keras.io/) - The neural networks API used
* [SKLearn](https://scikit-learn.org/stable/documentation.html) - Pre-processing 
* [OpenCV](https://opencv.org/) - Image Processing Library used
* [DATASET](datasetlink)
* [Pre-trained Model](Link)

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Author

* **Vishnu S Dev** - *Initial work* - [sherlocked777](https://github.com/sherlocked777)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

