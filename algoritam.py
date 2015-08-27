#Algoritam za prepoznavanje karakterističnih točaka lica (samostalni program)
#{v0.1}
"""
OPIS ALGORITMA
Algoritam koristi poznati Viola-Jones detektor

KONVENCIJE 
- Imena objekata (varijabli) vrlo su opisna zbog lakšeg snalaženja, a iz istog razloga su i na engleskom jeziku zbog boljeg uklapanja u OpenCV procedure te iz razloga što će algoritam biti postavljen na javni repozitorij
- Postavke algoritma vrlo se lako mogu konfigurirati u dijelu 'KONFIGURACIJA DETEKTORA', kao npr. korištenje različitih klasifikatora i parametara, itd.
- Algoritam je podjeljen na blokove (funkcije) radi lakšeg održavanja i izmjene
- Na kraju svake naredbe stavljen je ';', iako to u Pythonu nije praksa, no mišljenja sam da ih je korisno koristiti zbog toga što jednoznačno određuju kraj naredbe (u slučaju da se koristi višeredna naredba) te omoguću lakše shvaćanje koda osobama koje nisu navikle na Python sintaksu

Autor: kgrlic@foi.hr
"""

import numpy as np
from sympy import Point, Line, mpmath
from mpmath import *
import base64
import cv2

#---KONFIGURACIJA-DETEKTORA----------------------------------------------------
#Postavljanje datoteka koje sadrže klasifikatore (Haarove kaskade)
faceCascadeFile = 'haarcascade_frontalface_alt.xml';
eyeCascadeFile = 'haarcascade_eye.xml';
mouthCascadeFile = 'Mouth.xml';
noseCascadeFile = 'Nariz.xml';

#Postavljanje daototeka (fotografija) na kojima želimo detektirati objekte
imageFile_1 = 'jan2.jpg';
#imageFile_2 = 'fotografija.png';
#---KONFIGURACIJA-DETEKTORA----------------------------------------------------

#Globalne varijable
imageObj = cv2.imread(imageFile_1);
imageGrayObj = cv2.cvtColor(imageObj, cv2.COLOR_BGR2GRAY);


def loadCascadeClassifiersObjects():
    faceCascadeObj = cv2.CascadeClassifier(faceCascadeFile);
    eyeCascadeObj = cv2.CascadeClassifier(eyeCascadeFile);
    mouthCascadeObj = cv2.CascadeClassifier(mouthCascadeFile);
    noseCascadeObj = cv2.CascadeClassifier(mouthCascadeFile);          
    cascadeObjects = {'face':faceCascadeObj, 'eye':eyeCascadeObj, 'mouth':mouthCascadeObj, 'nose':noseCascadeObj};
    return cascadeObjects;

def detectFaces(cascadeObj, imgObj):
    detectedFacesObj = cascadeObj.detectMultiScale(imgObj, 1.3, 5);
    return detectedFacesObj;

def detectEyes(cascadeObj, regionOfInterestGrayObjects):
    detectedEyesObjects = [];
    for roiGrayObj in regionOfInterestGrayObjects:
        detectedEyesObjects.append(cascadeObj.detectMultiScale(roiGrayObj, 1.3, 5));
    return detectedEyesObjects;

def detectMouths(cascadeObj, regionOfInterestGrayObjects):
    detectedMouthsObjects = [];
    for roiGrayObj in regionOfInterestGrayObjects:
        detectedMouthsObjects.append(cascadeObj.detectMultiScale(roiGrayObj, 1.3, 5));
    return detectedMouthsObjects;

def detectNose(cascadeObj, regionOfInterestGrayObjects):
    detectedNoseObjects = [];
    for roiGrayObj in regionOfInterestGrayObjects:
        detectedNoseObjects.append(cascadeObj.detectMultiScale(roiGrayObj, 1.3, 5));
    return detectedNoseObjects;

def regionsOfInterestBottom(detectedObj, imageObj):
    regionOfInterestObjects = [];    
    for (rectX, rectY, rectWidth, rectHeight) in detectedObj:
        regionOfInterestObjects.append(imageObj[rectY + rectHeight*0.65:rectY + rectHeight,
        rectX:rectX + rectWidth]);
    return regionOfInterestObjects;

def regionsOfInterest(detectedObj, imageObj, hAdd = 0, hDiv = 1, wAdd = 0, wDiv = 1):
    regionOfInterestObjects = [];    
    for (rectX, rectY, rectWidth, rectHeight) in detectedObj:
        regionOfInterestObjects.append(imageObj[rectY + rectHeight*hAdd:rectY + rectHeight/hDiv,
        rectX + rectWidth*wAdd:rectX + rectWidth/wDiv]);
    return regionOfInterestObjects;

def calculateEyesPoints(detectedEyesObjects):
    eyeCenterPoints = [];
    for detectedEyesObj in detectedEyesObjects:
        for (rectX,rectY,rectWidth,rectHeight) in detectedEyesObj:
            eyeCenterPoints.append((rectX + rectWidth/2, rectY + rectHeight/2));
    return eyeCenterPoints;

def calculateEyebrowPoints(detectedEyesObjects):
    leftEye = True;
    eyebrowPoints = [];
    for detectedEyesObj in detectedEyesObjects:
        for (rectX,rectY,rectWidth,rectHeight) in detectedEyesObj:
            if(leftEye):
                leftEye = False;
                eyebrowPoints.append((rectX + rectWidth/10*(-2), rectY));
                eyebrowPoints.append((rectX + rectWidth, rectY));
            else:
                eyebrowPoints.append((rectX, rectY));
                eyebrowPoints.append((rectX + rectWidth/10*12, rectY));
    return eyebrowPoints;

def calculateEyeCornerPoints(detectedEyesObjects):
    leftEye = True;
    eyeCornerPoints = [];    
    for detectedEyesObj in detectedEyesObjects:    
        for (rectX,rectY,rectWidth,rectHeight) in detectedEyesObj:
            if(leftEye):
                leftEye = False;
                eyeCornerPoints.append((rectX, rectY + rectHeight/2));
                eyeCornerPoints.append((rectX + rectWidth, rectY + rectHeight/10*7));
            else:
                eyeCornerPoints.append((rectX, rectY + rectHeight/10*7));
                eyeCornerPoints.append((rectX + rectWidth, rectY + rectHeight/2));                
    return eyeCornerPoints;

def calculateMouthPoints(detectedMouthObjects):
    mouthPoints = [];
    for detectedMouthObj in detectedMouthObjects:
        (rectX,rectY,rectWidth,rectHeight) = detectedMouthObj[0];
        mouthPoints.append((rectX, rectY + rectHeight/4));
        mouthPoints.append((rectX + rectWidth, rectY + rectHeight/4));
    return mouthPoints;

def calculateNosePoints(detectedNoseObjects):
    nosePoints = [];
    for detectedNoseObj in detectedNoseObjects:
        for (rectX,rectY,rectWidth,rectHeight) in detectedNoseObj:
            nosePoints.append((rectX + rectWidth/2, rectY + rectHeight/2));
    return nosePoints;

def drawPoints(calculatedPoints, regionOfInterestColorObjects, color = (0,0,255)):
    for regionOfInterestColorObj in regionOfInterestColorObjects:
        for (x,y) in calculatedPoints:        
            cv2.circle(regionOfInterestColorObj,(x,y), 5, color, -1);
    return;

def calculateFaceTilt(faceEyesIrisPoints, regionOfInterestColorObjects):
    zeroLine = Line (Point (1,0), Point (0,0));
    leftEye = Point (faceEyesIrisPoints.pop());
    rightEye =  Point (faceEyesIrisPoints.pop());
    eyeMiddlePoint = Point ((leftEye + rightEye)/2);       
    eyeLine = Line (leftEye, rightEye);
    faceSymmetryLine = eyeLine.perpendicular_line(eyeMiddlePoint);
    angle = mpmath.degrees(eyeLine.angle_between(zeroLine));
    if (int(angle) > 90):    
        return {'angle':int(angle) - 180, 'tiltline':faceSymmetryLine};
    else:
        return {'angle':int(angle), 'tiltline':faceSymmetryLine};

def rotateImage(imageObj, correctionAngle):
    imageCenter = tuple(np.array(imageObj.shape[0:2])/2)
    rotationMatrix = cv2.getRotationMatrix2D(imageCenter, correctionAngle,1.0)
    return cv2.warpAffine(imageObj, rotationMatrix,(imageObj.shape[1], imageObj.shape[0]) ,flags=cv2.INTER_LINEAR)


cascadeObjects = loadCascadeClassifiersObjects();

#Detektiranje lica i roi
detectedFaces = detectFaces(cascadeObjects['face'], imageGrayObj);
regionOfInterestGrayObjects = regionsOfInterest(detectedFaces, imageGrayObj);
regionOfInterestColorObjects = regionsOfInterest(detectedFaces, imageObj);

#Detektiranje očiju i točaka očiju
detectedEyesObjects = detectEyes(cascadeObjects['eye'], regionOfInterestGrayObjects);
eyesCenterPoints = calculateEyesPoints(detectedEyesObjects);

#Detekcija kuta korekcije i pravca nagiba
calculatedFaceTiltObj = calculateFaceTilt(eyesCenterPoints, regionOfInterestGrayObjects);
correctionAngle = calculatedFaceTiltObj['angle'];
tiltLine = calculatedFaceTiltObj['tiltline'];
correctedImageObj = rotateImage(imageObj, calculatedFaceTiltObj['angle']);
correctedImageGrayObj = cv2.cvtColor(correctedImageObj, cv2.COLOR_BGR2GRAY);

#Detectiranje lica na koretiranoj slici
detectedFacesCorr = detectFaces(cascadeObjects['face'], correctedImageGrayObj);
regionOfInterestGrayObjectsCorr = regionsOfInterest(detectedFaces, correctedImageGrayObj);
regionOfInterestColorObjectsCorr = regionsOfInterest(detectedFaces, correctedImageObj);

#Detektiranje očiju i točaka očiju kod korektiranih slika
detectedEyesObjectsCorr = detectEyes(cascadeObjects['eye'], regionOfInterestGrayObjectsCorr);
eyesCenterPointsCorr = calculateEyesPoints(detectedEyesObjectsCorr);
eyebrowPointsCorr = calculateEyebrowPoints(detectedEyesObjectsCorr);
eyeCornerPointsCorr = calculateEyeCornerPoints(detectedEyesObjectsCorr);
drawPoints(eyesCenterPointsCorr, regionOfInterestColorObjectsCorr);
drawPoints(eyebrowPointsCorr, regionOfInterestColorObjectsCorr);
drawPoints(eyeCornerPointsCorr, regionOfInterestColorObjectsCorr);

#ROI za detektiranje točaka u donjem dijelu lica
regionOfInterestBottomColorObjectsCorr = regionsOfInterest(detectedFacesCorr, correctedImageObj, 0.65);
regionOfInterestBottomGrayObjectsCorr = regionsOfInterest(detectedFacesCorr, correctedImageGrayObj, 0.65);

#Detektiranje usta
detectedMouthsObjectsCorr = detectMouths(cascadeObjects['mouth'], regionOfInterestBottomGrayObjectsCorr);
mouthCornerPoints = calculateMouthPoints(detectedMouthsObjectsCorr);
drawPoints(mouthCornerPoints, regionOfInterestBottomColorObjectsCorr);

#ROI za detektiranje točaka u središnjem dijelu lica
regionOfInterestMiddleColorObjectsCorr = regionsOfInterest(detectedFacesCorr, correctedImageObj, 0.45, 1.4, 0.33, 1.4);
regionOfInterestMiddleGrayObjectsCorr = regionsOfInterest(detectedFacesCorr, correctedImageGrayObj, 0.45, 1.4, 0.33, 1.4);

#Detektiranje nosa
detectedNoseObjectsCorr = detectNose(cascadeObjects['nose'], regionOfInterestMiddleGrayObjectsCorr);
nosePointsCorr = calculateNosePoints(detectedNoseObjectsCorr);
drawPoints(nosePointsCorr, regionOfInterestMiddleColorObjectsCorr, (255,0,255));

   
cv2.imwrite('rotatedImge.jpg', imageObj);
cv2.imshow('image',regionOfInterestMiddleGrayObjectsCorr[0])
cv2.imshow('rotatedImge', correctedImageObj);
cv2.waitKey(0);
cv2.destroyAllWindows();
