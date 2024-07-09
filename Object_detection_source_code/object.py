import cv2
from google.colab.patches import cv2_imshow
img = cv2.imread('/content/cat.jpg')
cv2_imshow(img)
cv2.waitKey(0)
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
  classNames = f.read().rstrip('\n').split('\n')
print(classNames)
configPath = '/content/ssd_mobilenet_v3_large_coco_2020_01_14 (1).pbtxt'
weightsPath = '/content/frozen_inference_graph (1).pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
classIds, confs, bbox = net.detect(img, confThreshold=0.7)
print(classIds, bbox)
for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x+w, y+h), color=(0,255,0), thickness=3)
    cv2.putText(img, classNames[classId-1], (x+10, y+30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
cv2_imshow(img)
