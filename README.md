# Facial-Expression-Recognition
Facial Emotion Recognition using CNN . The dataset used is <a href="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data">fer2013</a>,
which was provided by kaggle as a challenge in 2013. It consist of a csv file which have pixel values of 
faces. We can detect 7 human (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) after sucessful 
training of the model.

The winner of this challenge had an accuracy of 71% ,whereas i was able to achieve 64% which is pretty close to it.

To train in your own dataset  use ```Facial_emotion_2.py``` file.

To check the accuracy of your trained model using the ```PrivateTest``` use file ```emotion_test```.

You can use my Trained model ```model_trained_64.p ``` to use in real time use ```emotion_analysis``` file 
