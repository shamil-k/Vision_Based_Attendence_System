# Vision_Based_Attendence_System


Face Recognition Based Attendence System Using diffrent fearures

Implementation:


1 - Face Recognition (MTCNN + Insight)


* Face Recognition /Identification/Verification

Using MTCNN:https://github.com/ipazc/mtcnn
we are detecting 

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
    }
]


* Face Detection 

From mxnet we are taking the implementation  insightface(Face Analysis Project on PyTorch and MXNet) from ArcFace Paper with mega face data set https://github.com/deepinsight/insightface


from https://arxiv.org/pdf/1801.07698.pdf ArcFace Research Paper


<img width="620" alt="Screenshot 2021-06-15 at 10 47 40 PM" src="https://user-images.githubusercontent.com/55822384/122096115-d3f69400-ce2b-11eb-8a49-aa0065da97b6.png">



Training a DCNN for face recognition supervised by the ArcFace loss. Based on the feature xi and weight W normalisation, we
get the cos θj (logit) for each class as WT
j xi. We calculate the arccosθyi
and get the angle between the feature xi and the ground truth
weight Wyi
. In fact, Wj provides a kind of centre for each class. Then, we add an angular margin penalty m on the target (ground truth)
angle θyi
. After that, we calculate cos(θyi + m) and multiply all logits by the feature scale s. The logits then go through the softmax
function and contribute to the cross entropy loss

2 - app.py

UI user interface with  tk-inter  


