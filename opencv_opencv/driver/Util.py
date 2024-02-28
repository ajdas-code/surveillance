import pprint
import cv2
import os, ffmpeg

class Util:
    # c'tor
    def __init__(self):
        pass

    # class method

    # compress video file to a target size
    @staticmethod
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
                      **{'c:v': 'libx264', 'b:v': video_bitrate,
                         'pass': 1, 'f': 'mp4'}).overwrite_output().run()
        ffmpeg.output(i, output_file_name,
                      **{'c:v': 'libx264', 'b:v': video_bitrate,
                         'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}).overwrite_output().run()
        return

    # compress video file to a target size
    # function to draw rectangle on image
    # according to given (x, y) coordinates and
    # given width and heigh
    @staticmethod
    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # function to draw text on give image starting from
    # passed (x, y) coordinates.
    @staticmethod
    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    # find the pre-trained HAAR model file used for detection
    # 'f' frontal face
    # 'e' eye
    # 'alt' alt frontal face
    # 'alt2' alt2 frontal face
    # 'alt_tree' alt tree frontal face
    # default frontal face
    @staticmethod
    def findHaarModel(arg):
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        switcher = {
            'f': "data/haarcascade_frontalface_default.xml",
            'e': "data/haarcascade_eye.xml",
            'alt': "data/haarcascade_frontalface_alt.xml",
            'alt2': "data/haarcascade_frontalface_alt2.xml",
            'alt_tree': "data/haarcascade_frontalface_alt_tree.xml",
        }
        haar_model = os.path.join(cv2_base_dir, switcher.get(arg, "data/haarcascade_frontalface_default.xml"))
        return haar_model
