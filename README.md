# Gender_Classifier_Face
## About
Repository of a Machine Learning Project in which the model predicts the gender of a person by checking for various facial features.

## Usage
You will need `python 3` on your system.

To check whether you have `python 3`
type the following command in command prompt `python -V`.
If you do not have `python 3`, 
Follow the installation guide [here](https://docs.python.org/3/using/windows.html)

All dependencies can be found in requirements.txt file.
To download the depecndencies from requirements.txt
type the following command in commpand prompt `pip install requirements.txt`

To Train the model:
+ Download the dataset from [face_dataset](https://drive.google.com/drive/folders/1oDE2ZqYwktugXhO79Zgo2dzrFdPLLyNq?usp=sharing)
+ Open the command prompt
+ Type the following command `python training_face.py`

To Test the model with images of your choice:
+ Navigate to the folder containing your python script and download the image (Example: example_image.jpg) into that folder or create a new folder in the same (Example: example_folder) and download the image there
+ Open the command prompt
+ Type the following command `python testing_images.py --image example_folder/example_image.jpg`

To Test the model with webcam:
+ Open the command prompt
+ Type the following command `python testing_webcam.py`
