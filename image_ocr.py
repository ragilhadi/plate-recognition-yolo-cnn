from recognition import *
from classification import *
import easyocr
import pyttsx3
import time
start_time = time.time()

test_image_file = 'test.jpg'

image = cv2.imread(test_image_file)
image = cv2.resize(image, frame_size)
frame, results = object_detection(image)
engine = pyttsx3.init()

font = cv.FONT_HERSHEY_SIMPLEX
org = (100, 50)
fontScale = 1
color = (0, 255, 0)
thickness = 2    

reader = easyocr.Reader(['en'])

def cleanup_text(text):
    cleanup_text = ''
    if len(text) > 0:
        cleanup_text = text[0][-2]
    return cleanup_text

if len(results) > 0:
    for result in results:
        x1,y1,x2,y2,cnf, clas = result
        crop = frame[y1:y2, x1:x2]
        ocr = reader.readtext(crop)
        text = cleanup_text(ocr)
        cv.putText(frame, text , org, font, fontScale, color, thickness, cv.LINE_AA)
        engine.say(text)
        engine.runAndWait()
    
    
    
cv2.imshow("Result OCR", frame)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('export/result_ocr.jpg', frame)







