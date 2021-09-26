import cv2
import numpy
import face_recognition
import os

path = "Images"
#create a list of all imported images
images = []
classnames = []
mylist = os.listdir(path)
#print (mylist)
#import images
for i in mylist:
    curImg=cv2.imread(f'{path}/{i}')
    if not i.startswith('.'): #ignore hidden files
        images.append(curImg)
        classnames.append(os.path.splitext(i)[0])
#print (classnames)


#Function that finds the encodings of all the images
def findEncodings (images):
    encodelist =[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode =face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodeListKnown=findEncodings(images)
#print ("Encoding complete")
#print (len(encodeListKnown))

#Initialise webcam
capture=cv2.VideoCapture(0)
# Find locations of all faces in the camera, encode them and compare them to the known encodings for
# complete facial recognition
while True:
    success, img = capture.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFram = face_recognition.face_locations(imgS)
    encodesCurFram = face_recognition.face_encodings(imgS,facesCurFram)

    #compare encodings
    for encodeFace,faceLoc in zip (encodesCurFram,facesCurFram):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print (matches)
        matchIndex = numpy.argmin(faceDist) #return index with minimum number in the list

        if matches[matchIndex]:
            name=classnames[matchIndex].upper()
            #print (name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 #since faceLoc is of the small camera image
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #cv2.rectangle(img,(x1,y2-35),(x2, y2),(0, 255, 0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow("Webcam",img) #Display original/larger image from the camera
    cv2.waitKey(1)