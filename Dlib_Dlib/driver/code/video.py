# Python program to illustrate
# saving an operated video
# organize imports
import face_recognition
import numpy as np
import cv2
import pprint
import sys
import time
import os, ffmpeg
TEMP_FILE = 'temp.mp4'

def compress_video(video_full_path, output_file_name, target_size):
    # Reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
    min_audio_bitrate = 32000
    max_audio_bitrate = 256000

    probe = ffmpeg.probe(video_full_path)
    # Video duration, in s.
    duration = float(probe['format']['duration'])
    # Audio bitrate, in bps.
    # >>> print(next((i) for i in list if i%2==0))
    # 2
    # >>>
    aud_val = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
    if aud_val:
        audio_bitrate = float(['bit_rate'])
    else:
        audio_bitrate = min_audio_bitrate
    # audio_bitrate = float(next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
    # Target total bitrate, in bps.
    target_total_bitrate = (target_size * 1024 * 8) / (1.073741824 * duration)

    # Target audio bitrate, in bps
    if 10 * audio_bitrate > target_total_bitrate:
        audio_bitrate = target_total_bitrate / 10
        if audio_bitrate < min_audio_bitrate < target_total_bitrate:
            audio_bitrate = min_audio_bitrate
        elif audio_bitrate > max_audio_bitrate:
            audio_bitrate = max_audio_bitrate
    # Target video bitrate, in bps.
    video_bitrate = target_total_bitrate - audio_bitrate

    i = ffmpeg.input(video_full_path)
    ffmpeg.output(i, os.devnull,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
                  ).overwrite_output().run()
    ffmpeg.output(i, output_file_name,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
                  ).overwrite_output().run()
    return


if __name__ == "__main__":
    # execute only if run as a script

    print("This is the name of the program:", sys.argv[0])
    print("Argument List:", sys.argv)
    if (len(sys.argv) < 3):
        pprint.pprint("{} : This program needs two args - <inputfile> and <outputfile>".format(sys.argv[0]))
        pprint.pprint("Usage: python3.6 {} <inputfile> <outputfile>".format(sys.argv[0]))
        exit()
    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(sys.argv[1])
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Define the codec and create VideoWriter object
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(TEMP_FILE, fourcc, 15.0, (int(w), int(h)))
    output_file_name = str(sys.argv[2])

    # Load some sample pictures and learn how to recognize them.
    lmm_image = face_recognition.load_image_file("img1.jpg")
    lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

    al_image = face_recognition.load_image_file("img3.jpg")
    al_face_encoding = face_recognition.face_encodings(al_image)[0]

    known_faces = [
        lmm_face_encoding,
        al_face_encoding
    ]

    known_names = [
        'Ajitesh Das',
        'Gouri Das'
    ]
    # Initialize some variables
    face_locations = []
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
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        # for each frame process face recognition on the all known faces
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            # If you had more than 2 faces, you could make this logic a lot prettier
            pprint.pprint(matches)
            for index in range(len(known_faces)):
                if matches[index-1]:
                    face_names.append(known_names[index-1])
            continue
        # Label the results on original frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            continue
        pprint.pprint("Append the frame to output file")
        # output the frame
        out.write(frame)
        localtime = time.localtime()
        result = time.strftime("%I:%M:%S %p", localtime)
        print("{}:>>>>Writing frame {} / {}".format(result, current_frame, length))
        continue

    # Close the window / Release webcam
    cap.release()
    # After we release our webcam, we also release the output
    out.release()
    # De-allocate any associated memory usage
    cv2.destroyAllWindows()
    pprint.pprint("Status: Generation of output movie....Done")
    pprint.pprint("Status: Trying to commpress output movie....Starting")
    compress_video(TEMP_FILE, output_file_name, 5 * 1000)
    pprint.pprint("Status: Trying to commpress output movie {}....Done".format(output_file_name))
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)
        pprint.pprint("Status: Cleaning up staging {} done....Done".format(TEMP_FILE))
    else:
        print("The file does not exist")
        pprint.pprint("Status: Error Cleaning up staging {} done....Done".format(TEMP_FILE))
    exit()
