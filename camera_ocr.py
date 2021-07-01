import cv2 as cv
from recognition import *
from classification import *
import tensorflow as tf
import easyocr
import pyttsx3


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


reader = easyocr.Reader(['en'])


print(f'{height}x{width} with {frame}FPS')
writer = cv.VideoWriter('export/videos/papan-ocr.mp4', cv.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

def cleanup_text(text):
    cleanup_text = ''
    if len(text) > 0:
        cleanup_text = text[0][-2]
        
    return cleanup_text

while True:
    isTrue, frame = capture.read()
    frameDetect, results = object_detection(frame)
    print(results)
    if len(results) > 0:
        for result in results:
            x1,y1,x2,y2,cnf, clas = result
            crop = frame[y1:y2, x1:x2]
            ocr = reader.readtext(crop)
            text = cleanup_text(ocr)
            cv.putText(frameDetect, text , org, font, fontScale, color, thickness, cv.LINE_AA)
            engine.say(text)
            engine.runAndWait()   

    cv.imshow('Video Capture', frameDetect)

    writer.write(frameDetect)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
writer.release()
cv.destroyAllWindows()
