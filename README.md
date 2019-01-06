# People-Identification-With-Keras
A Deep Learning model that recognizes people from a set of images
Repository contains:
  1-pre_processing.ipynb: Preprocess the dataset for further training
  2-generate_model.ipynb: Used to generate model
  3-prediction.ipynb: Prediction script for recognizing people
  base_model_vgg.py: Model script(Here I used Mini VGGNet)
Dataset
  Dataset contains more than 5k people images : Link:
------------
How to Run
------------
To run the code follow these steps:
1. Directory structure:
	People-Identification-With-Keras
		---Dataset:Contain sub folders, each of different person images
			---Aaron_Eckhart
				---Aaron_Eckhart_0001.jpg
				---Aaron_Eckhart_0002.jpg
				--- ----
				--- ----
			---Aaron_Guiel
				---Aaron_Guiel_0001.jpg
				---Aaron_Guiel_0002.jpg
				--- ----
				--- ----
			--- -----
			--- -----
		---1-1-pre_processing.ipynb
		---2-generate_model.ipynb
		---3-prediction.ipynb
		---base_model_vgg.py
		---README.md
		---LICENSE
2. Activate virtual env(python3)
3. RUN pip install -r requirement.txt
4. Run 1-1-pre_processing.ipynb for preprocess images
	This will create train and test meta data files for further processing
	This will also create 'class_labels.csv', that contains target labels which we use later while prediction
5. RUN 2-generate_model.ipynb
	This will generate model named 'model.h5' in same directory
	Trains on MiniVGGNet(base_model.py)
	User can tune the hyper parameters (epochs, batch size, optimizer, weight decay, etc..)
6. RUN 3-prediction.ipynb
	This will takke random images from dataset directory for prediction
	model will predict person name
	dlib library used for detecting face

