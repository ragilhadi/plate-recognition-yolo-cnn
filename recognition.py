import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from torchvision import models
from torchvision import transforms
from PIL import Image
import numpy as np
import time


yolov5_weight_file = 'models_weight/papan.pt'
conf_set=0.35 
frame_size=(800, 480)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolov5_weight_file, map_location=device)
cudnn.benchmark = True 
names = model.module.names if hasattr(model, 'module') else model.names

transform = transforms.Compose([
			transforms.Resize(144),
			# transforms.CenterCrop(142),
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])
		  ]) 

def object_detection(frame):
	img = torch.from_numpy(frame)
	img = img.permute(2, 0, 1).float().to(device)
	img /= 255.0  
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	pred = model(img, augment=False)[0]
	pred = non_max_suppression(pred, conf_set, 0.30) # prediction, conf, iou

	detection_result = []
	for i, det in enumerate(pred):
		if len(det): 
			for d in det: # d = (x1, y1, x2, y2, conf, cls)
				x1 = int(d[0].item())
				y1 = int(d[1].item())
				x2 = int(d[2].item())
				y2 = int(d[3].item())
				conf = round(d[4].item(), 2)
				c = int(d[5].item())
				
				detected_name = names[c]

				print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}')
				detection_result.append([x1, y1, x2, y2, conf, c])
				
				frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1) # box
				if c!=1: # if it is not head bbox, then write use putText
					frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

	return (frame, detection_result)




