# Python program to illustrate
# saving an operated video
# organize imports
import numpy as np
import cv2
import pprint
import sys
import time
import os
from PIL import Image
import Util
import pandas as pd

TEMP_FILE = 'temp.mp4'

if __name__ == "__main__":
    # execute only if run as a script
    # ------------------------------

    # Step#1: Process the input of the program
    #         Required - 4 imputs - input video, output video, training datafile and label csv datasource
    print("This is the name of the program:", sys.argv[0])
    print("Argument List:", sys.argv)
    if (len(sys.argv) < 5):
        pprint.pprint("{} : This program needs four args - <inputfile>, <outputfile>, training datafile, label_csv_file".format(sys.argv[0]))
        pprint.pprint("Usage: python3.6 {} <inputfile> <outputfile> <training_data_file> <label_csv_file>".format(sys.argv[0]))
        exit()
    input_file_name = str(sys.argv[1])
    output_file_name = str(sys.argv[2])
    training_data_file = str(sys.argv[3])
    label_csv_file = str(sys.argv[4])
    # ------------------------------

    # Step#2: Setting up input file and output file
    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(input_file_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Define the codec and create VideoWriter object
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Output
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(TEMP_FILE, fourcc, 15.0, (int(w), int(h)))
    pprint.pprint("Statistics: # of frames: {0} ; w : {1}, h : {2}".format(length, w, h))
    # ------------------------------

    # Step 3: using training_data_file and label_csv_file
    # Load training data .
    face_cascade = cv2.CascadeClassifier(Util.Util.findHaarModel('f'))
    face_recognizer = cv2.face_LBPHFaceRecognizer.create()
    # face_recognizer.load(training_data_file)
    face_recognizer.read(training_data_file)
    df = pd.read_csv(label_csv_file, index_col=0)
    d = df.to_dict("split")
    label_dictionary = dict(zip(d["index"], d["data"]))
    pprint.pprint("Label data = {}".format(label_dictionary))
    # done data loads
    # ------------------------------

    # Step 4: Process each frame in iteration as image
    # Initialize some variables
    face_encodings = []
    face_names = []
    frame_number = 0
    # loop runs if capturing has been initialized.
    current_frame = 0
    while(True):
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1
        # Converts to RGB color space, OCV reads colors as BGR
        # frame is converted to RGB to do further processing
        # of face detections
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find all the faces and face encodings in the current frame of video
        face_encodings = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                                       minNeighbors=5, minSize=(20, 20),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        # face_encodings = face_cascade.detectMultiScale(gray_frame)
        # face_encodings = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        # for each frame process face recognition on the all known faces
        pprint.pprint("No of faces found: {}".format(len(face_encodings)))
        img_crop = []
        for (x, y, w, h) in face_encodings:
            # Convert Face to greyscale
            roi_gray = gray_frame[y:y+h, x:x+w]
            # recognize the Face
            id_, conf = face_recognizer.predict(roi_gray)
            pprint.pprint("-->id_, conf : {}, {}".format(id_, conf))
            # Font style for the name
            font = cv2.FONT_HERSHEY_SIMPLEX
            if conf < 200 :
                # Get the name from the List using ID number
                pprint.pprint("id_ : {}".format(id_))
                name = label_dictionary.get(str(id_), "Unknown")
                pprint.pprint("Label data = {}".format(label_dictionary))
                cv2.putText(frame, name[0], (x, y), font, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # end if
            else:
                pprint.pprint("Unknown face found!!!********************************")
                cv2.putText(frame, "Unknown", (x, y), font, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        pprint.pprint("Append the frame to output file")
        # output the frame
        out.write(frame)
        localtime = time.localtime()
        result = time.strftime("%I:%M:%S %p", localtime)
        print("{}:>>>>Writing frame {} / {}".format(result, current_frame, length))
        # time.sleep(1)
        continue

    # Close the window / Release webcam
    cap.release()
    # After we release our webcam, we also release the output
    out.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
    pprint.pprint("Status: Generation of output movie....Done")
    pprint.pprint("Status: Trying to commpress output movie....Starting")
    Util.Util.compress_video(TEMP_FILE, output_file_name, 5 * 1000)
    pprint.pprint("Status: Trying to commpress output movie {}....Done".format(output_file_name))
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
        pprint.pprint("Status: Cleaning up staging {} done....Done".format(TEMP_FILE))
    else:
        print("The file does not exist")
        pprint.pprint("Status: Error Cleaning up staging {} done....Done".format(TEMP_FILE))
    exit()
