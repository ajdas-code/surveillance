# Python program to illustrate
# saving an operated video
# organize imports
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
    # loop runs if capturing has been initialized.
    current_frame = 0
    while(True):
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1
        # Converts to HSV color space, OCV reads colors as BGR
        # frame is converted to hsv
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # output the frame
        out.write(hsv)
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
