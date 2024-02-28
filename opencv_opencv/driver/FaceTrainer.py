# FaceTrainer
import cv2
import pprint
import sys
import time
import os
from PIL import Image
from Util import Util
import numpy as np
import math
import pandas as pd
from tempfile import gettempdir
from shutil import rmtree


class FaceTrainer:
    def __init__(self, data_folder_path, thumbnail_dir):
        self.data_folder_path = data_folder_path
        self.mapper = {}
        self.mapper['thumbnail'] = thumbnail_dir

    def getmapper(self):
        print("getter method called")
        return self.mapper

    @staticmethod
    def convertToNumber (s):
        return int.from_bytes(s.encode(), 'little')

    @staticmethod
    def convertFromNumber (n):
        return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

    def detect_single_face_for_taining(self, img_path):
        # read image
        img = cv2.imread(img_path)

        # convert the test image to gray scale as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # load OpenCV face detector, I am using LBP which is fast
        # there is also a more accurate but slow: Haar classifier
        face_cascade = cv2.CascadeClassifier(Util.findHaarModel('f'))
        # let's detect multiscale images(some images may be closer to camera than others)
        # result is a list of faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # if no faces are detected then return original img
        if (len(faces) == 0):
            return None, None

        # under the assumption that there will be only one face,
        # extract the face area
        (x, y, w, h) = faces[0]

        # return only the face part of the image
        return (gray[y:y+w, x:x+h], faces[0])

    def alt_detect_single_face_for_taining(self, img_path):
        # using pillow
        # convert the test image to gray scale as opencv face detector expects gray images
        Gery_Image = Image.open(img_path).convert("L")
        # Crop the Grey Image to 550*550 (Make sure your face is in the center in all image)
        Crop_Image = Gery_Image.resize( (550, 550) , Image.ANTIALIAS)
        gray = np.array(Crop_Image, "uint8")
        # load OpenCV face detector, I am using LBP which is fast
        # there is also a more accurate but slow: Haar classifier
        face_cascade = cv2.CascadeClassifier(Util.findHaarModel('f'))
        # let's detect multiscale images(some images may be closer to camera than others)
        # result is a list of faces
        faces = face_cascade.detectMultiScale(gray,  scaleFactor=1.5, minNeighbors=5)

        # if no faces are detected then return original img
        if (len(faces) == 0):
            return None, None

        # under the assumption that there will be only one face,
        # extract the face area
        (x, y, w, h) = faces[0]

        # return only the face part of the image
        return (gray[y:y+w, x:x+h], faces[0])

    # this function will read all persons' training images, detect face from each image
    # and will return two lists of exactly same size, one list
    # of faces and another list of labels for each face
    def prepare_training_data(self):

        # ------STEP-1--------
        # get the directories (one directory for each subject) in data folder
        dirs = os.listdir(self.data_folder_path)

        # list to hold all subject faces
        faces = []
        # list to hold labels for all subjects
        labels = []
        label = -1
        # let's go through each directory and read images within it
        for dir_name in dirs:
            # our subject directories start with letter 's' so
            # ignore any non-relevant directories if any
            if not dir_name.startswith("s"):
                continue

            # ------STEP-2--------
            # extract label number of subject from dir_name
            # format of dir name = slabel
            # , so removing letter 's' from dir_name will give us label
            label = label+1
            value = dir_name[1:]
            pprint.pprint("Label: {} , value : {}".format(label, value))
            # build path of directory containing images for current subject subject
            # sample subject_dir_path = "training-data/s1"
            subject_dir_path = self.data_folder_path + "/" + dir_name

            # get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            # ------STEP-3--------
            # go through each image name, read image,
            # detect face and add face to list of faces
            for image_name in subject_images_names:

                # ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue
                # for not image files ending with jpeg,jpg or png
                if ( not ( image_name.lower().endswith("jpeg") or
                           image_name.lower().endswith("jpg") or
                           image_name.lower().endswith("png"))):
                    continue

                # build image path
                # sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name
                print("Extracting image : {}".format(image_path))

                # detect face
                (face, rect) = self.detect_single_face_for_taining(image_path)

                # ------STEP-4--------
                # for the purpose of this tutorial
                # we will ignore faces that are not detected
                if face is not None:
                    # add face to list of faces
                    faces.append(face)
                    # add label for this face
                    labels.append(label)
                    # append to mapper
                    self.mapper[label] = value

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        return faces, labels


if __name__ == "__main__":
    # execute only if run as a script

    print("This is the name of the program:", sys.argv[0])
    print("Argument List:", sys.argv)
    if (len(sys.argv) != 4):
        pprint.pprint("{} : This program needs three args -image folder path, training file and mapping csv file".format(sys.argv[0]))
        pprint.pprint("Usage: python3.6 {} training out.yml mapping.csv".format(sys.argv[0]))
        exit()
    print("Initializing Face Trainer")
    tmp = os.path.join(gettempdir(), '.{}'.format(hash(os.times())))
    rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    pprint.pprint("Tempdir created ...{}".format(tmp))
    ft = FaceTrainer(sys.argv[1], tmp)
    outputfile = sys.argv[2]
    mappingfile = sys.argv[3]
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    print("Initializing Face Training process...")
    (faces, labels) = ft.prepare_training_data()
    print("Face Training process completed...")
    # print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    pprint.pprint("labels - {}".format(labels))
    print("Mapping - {}".format(ft.mapper))
    print("Calibrating face training data...")
    recognizer.train(faces, np.array(labels))
    print("Writing face training data to {}...".format(outputfile))
    recognizer.save(outputfile)
    print("Saving the mapping data for face id and name")
    df = pd.DataFrame.from_dict(ft.mapper, orient="index")
    df.to_csv(mappingfile)
    print("done...")
    exit()
