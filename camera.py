import cv2 as cv
from recognition import *
from classification import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3


test_image_file = 'images/picture (3).jpg'
cnn_weight_file = 'models_weight/charachter-plate-new.h5'
model = load_model(cnn_weight_file, compile=False)
engine = pyttsx3.init()

capture = cv.VideoCapture(0)
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
frame = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

font = cv.FONT_HERSHEY_SIMPLEX
org = (100, 50)
fontScale = 1
color = (0, 255, 0)
thickness = 2    

print(f'{height}x{width} with {frame}FPS')
writer = cv.VideoWriter('export/videos/papan.mp4', cv.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

while True:
    isTrue, frame = capture.read()
    frameDetect, results = object_detection(frame)
    print(results)
    if len(results) > 0:
        for result in results:
            x1,y1,x2,y2,cnf, clas = result
            crop = frame[y1:y2, x1:x2]
            text = process_segmentation_char(crop, model)
            cv.putText(frameDetect, text , org, font, fontScale, color, thickness, cv.LINE_AA)
            engine.say(text)
            engine.runAndWait()     

    cv.imshow('Mirror Video Capture', frameDetect)

    writer.write(frameDetect)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
writer.release()
cv.destroyAllWindows()
