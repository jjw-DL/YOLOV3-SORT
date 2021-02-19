import numpy as np
import cv2
import matplotlib.pyplot as plt

class YOLO(object):
    _defaults = {
        "labelsPath" : "./yolo-coco/coco.names",
        "weightsPath" : "./yolo-coco/yolov3.weights",
        "configPath" : "./yolo-coco/yolov3.cfg",
        "className" : "car",
        "threshold": 0.3,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.labelsName = self._get_labels()
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.output = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.W = None
        self.H = None
        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')

    def _get_labels(self):
        return open(self.labelsPath).read().strip().split('\n')

    def predict(self, frame):
        # 读取图像
        self.H, self.W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True , crop = False)
        # 将blob输入到前向网络中，并进行预测
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.output)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = detection[4] * scores[classID]

                if confidence > self.threshold:
                    # 将边界框的坐标还原至与原图片匹配，YOLO返回的是边界框的中心坐标以及边界框的宽度和高度
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # 计算左上角坐标
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)]) #左上角 + 长宽
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        dets = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                if self.labelsName[classIDs[i]] == self.className:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # color = [int(c) for c in self.COLORS[i % len(self.COLORS)]]
                    # cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    dets.append([x, y, x + w, y + h, confidences[i]])
        
        # plt.imshow(frame[:,:,::-1])
        # plt.show()
        return np.array(dets)


# if __name__ == "__main__":
#     yolo = YOLO()
#     frame = cv2.imread("./images/car1.jpg")
#     print(yolo.predict(frame))