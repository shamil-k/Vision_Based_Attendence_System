# Vision_Based_Attendence_System


Face Recognition Based Attendence System Using diffrent fearures

Implementation:



app.py

we were given 3 button in our UI



* Take Image


     > By clicking the take image button calling the function CollectImageForRegistration() > CollectimagesFromCamera()
     > 
     > Here It will collect frame by frame  default 50 images. It only use when ever the new user is coming or directly go with Predict image
     > Creating a folder for new user
     > 
     > giving the images name as micro second
     > 
     > Get all faces of current frame
     > 
     > Giving the condition to Get only biggest faces
     > 
     > Creating the bounding box for the face
     > 
     > Get each of the 3 frames Detecting the landmark of the faces using MTCNN
     > 
     > Resizing the Image size as 112,112
     > 
     > Saving the images in data set folder with names
     > 
     > Finally after taking the images of face "Image Captured"


* Train Image

     Data preprocessing:

     > Generate Face Embedding class 
     > quantifying faces - grab the path to the input image in our data set
     > Initialize the face embedder  - Generating 128 co-ordinate from faces using insightface model
     > Initialize list of extracted facial embedding and corresponding people names
     > Saving embedding faces in a single  pickle file

     To Train the model:

     > Features : Embedding of the faces
     > 
     > Target. : classification of the users

     Train keras model for Face Recognition:

     > Encode the labels for the names(Target)
     > 
     > Build Softmax classifier
     > 
     > Training the model
     > s
     > Saving the face recognition model 


* Predict Face

    > We need to compare the predicted face to actual face for that following steps:

    > Crop the faces from the frame
    > 
    > Convert into embedding
    > 
    > Comparing this embedding from  Orgianl embedding of the trained user faces
    > 
    > Checking this value is coming close to which cluster from Face Recognition model





Face Recognition (MTCNN + Insight)


>  Face Identification

Using MTCNN:https://github.com/ipazc/mtcnn it trained on  mega face data set

we are detecting faces

 >  
 >  
      detector = MTCNN()
      detector.detect_faces(img)
    [


    {
    
        'box': [277, 90, 48, 63],
        'keypoints':
        {
            'nose': (303, 131),
            'mouth_right': (313, 141),
            'right_eye': (314, 114),
            'left_eye': (291, 117),
            'mouth_left': (296, 143)
        },
        'confidence': 0.99851983785629272
      }]


>  Face Detection 

From mxnet we are taking the implementation  insightface(Face Analysis Project on PyTorch and MXNet) from ArcFace Paper  https://github.com/deepinsight/insightface


from https://arxiv.org/pdf/1801.07698.pdf ArcFace Research Paper




<img width="620" alt="Screenshot 2021-06-15 at 10 47 40 PM" src="https://user-images.githubusercontent.com/55822384/122096115-d3f69400-ce2b-11eb-8a49-aa0065da97b6.png">




