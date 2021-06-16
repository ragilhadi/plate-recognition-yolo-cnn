from recognition import *
from classification import *
import tensorflow as tf
from tensorflow.keras.models import load_model

test_image_file = 'test.jpg'
cnn_weight_file = 'models_weight/charachter-plate.h5'
model = load_model(cnn_weight_file, compile=False)

image = cv2.imread(test_image_file)
image = cv2.resize(image, frame_size)
frame, results = object_detection(image)

for result in results:
    x1,y1,x2,y2,cnf, clas = result
    crop = frame[y1:y2, x1:x2]
    text = process_segmentation_char(crop, model)
    
    
appendfont = cv.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2    
cv.putText(frame, text , org, font, fontScale, color, thickness, cv.LINE_AA)       
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
#cv2.imwrite('result.jpg', image_convert)





