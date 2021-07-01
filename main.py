from recognition import *
from classification import *
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3
import time
start_time = time.time()

test_image_file = 'images/picture (18).jpg'
cnn_weight_file = 'models_weight/charachter-plate-new.h5'
model = load_model(cnn_weight_file, compile=False)
engine = pyttsx3.init()

image = cv2.imread(test_image_file)
image = cv2.resize(image, frame_size)
frame, results = object_detection(image)

font = cv.FONT_HERSHEY_SIMPLEX
org = (100, 50)
fontScale = 1
color = (0, 255, 0)
thickness = 2    

if len(results) > 0:
    for result in results:
        x1,y1,x2,y2,cnf, clas = result
        crop = frame[y1:y2, x1:x2]
        text = ''
        text = process_segmentation_char(crop, model)
        engine.say(text)
        engine.runAndWait()
        cv.putText(frame, text , org, font, fontScale, color, thickness, cv.LINE_AA)   
    
    
    
cv2.imshow("Result", frame)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('export/test/cnn/picture (18).jpg', frame)
    







