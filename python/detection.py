import sys
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

sys.path.append(os.path.join(os.getcwd(), 'yolov5'))
sys.path.append(os.path.join(os.getcwd(), 'yolov5', 'utils'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

class ObjectDetector:
    def __init__(self, weights_path, roi_dangerzone, roi_restrictzone):
        device = select_device('0' if torch.cuda.is_available() else 'cpu')

        if isinstance(weights_path, (Path, str)):
            weights_path = str(weights_path)
        
        self.model = DetectMultiBackend(weights_path, device=device)
        self.model.warmup(imgsz=(1, 3, 640, 640))

        self.roi_dangerzone = roi_dangerzone
        self.roi_restrictzone = roi_restrictzone
        self.tracked_objects = {}

    def detect_and_draw(self, frame):
        img = letterbox(frame, new_shape=(640, 640))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.model.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)
        pred = non_max_suppression(pred)

        detected_events = []
        current_time = time.time()

        # Track objects
        objects = {}  # To store detected objects

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    class_name = self.model.names[int(cls)]
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    object_id = f"{int(center_x)}-{int(center_y)}"

                    # 배회 감지 로직 -> 삭제
                    # if class_name == 'person':
                    #     object_id = f"{int(center_x)}-{int(center_y)}"
                    #     if object_id in self.tracked_objects:
                    #         time_in_roi = current_time - self.tracked_objects[object_id]['start_time']
                    #         if time_in_roi > self.loitering_time_threshold:
                    #             detected_events.append({'type': '배회', '신뢰도': conf.item()})
                    #             cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 3)
                    #             cv2.putText(frame, 'Loitering', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    #     else:
                    #         self.tracked_objects[object_id] = {'start_time': current_time}
                
                    if class_name == 'person':
                        objects[object_id] = {'class': class_name, 'center': (center_x, center_y), 'bbox': (xyxy[0], xyxy[1], xyxy[2], xyxy[3]), 'conf': conf}
                    elif class_name == 'box':
                        objects[object_id] = {'class': class_name, 'center': (center_x, center_y), 'bbox': (xyxy[0], xyxy[1], xyxy[2], xyxy[3]), 'conf': conf}
                    elif class_name == 'forklift':
                        objects[object_id] = {'class': class_name, 'center': (center_x, center_y), 'bbox': (xyxy[0], xyxy[1], xyxy[2], xyxy[3]), 'conf': conf}
                    
                    
                    # 침입 감지 로직 intrusion -> 위험 구역 dangerzone
                    if class_name == 'person' and \
                        self.roi_dangerzone[0] < center_x < self.roi_dangerzone[0] + self.roi_dangerzone[2] and \
                        self.roi_dangerzone[1] < center_y < self.roi_dangerzone[1] + self.roi_dangerzone[3]:
                        detected_events.append({'type': '위험 구역', '신뢰도': conf.item()})
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 3)
                        cv2.putText(frame, 'danger zone', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    # 불법 주차 감지 로직 no_parking ->  제한구역
                    elif class_name == 'forklift' and \
                        self.roi_restrictzone[0] < center_x < self.roi_restrictzone[0] + self.roi_restrictzone[2] and \
                        self.roi_restrictzone[1] < center_y < self.roi_restrictzone[1] + self.roi_restrictzone[3]:
                        detected_events.append({'type': '제한 구역', '신뢰도': conf.item()})
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 3)
                        cv2.putText(frame, 'restrict zone', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    else:
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 차와 사람의 거리가 가까울 때 알림
        for obj_id1, obj1 in objects.items():
            if obj1['class'] == 'person':
                for obj_id2, obj2 in objects.items():
                    if obj2['class'] == 'forklift':
                        distance = np.sqrt((obj1['center'][0] - obj2['center'][0])**2 + (obj1['center'][1] - obj2['center'][1])**2)
                        if distance < 300:
                            detected_events.append({'type': '근접 알림', '신뢰도': min(obj1['conf'].item(), obj2['conf'].item())})
                            cv2.line(frame, (int(obj1['center'][0]), int(obj1['center'][1])), (int(obj2['center'][0]), int(obj2['center'][1])), (255, 255, 0), 2)
                            cv2.putText(frame, 'Proximity Alert', (int((obj1['center'][0] + obj2['center'][0]) / 2), int((obj1['center'][1] + obj2['center'][1]) / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)



        return frame, detected_events
