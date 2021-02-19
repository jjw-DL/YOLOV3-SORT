# YOLOV3-SORT实现车辆跟踪与车流统计
- 代码简单，适合新手学习，ID Switch较多。
- 先学习SORT算法，明白基于检测的目标跟踪基本框架，然后在学习Deepsort。
- Tutorial文件夹是教程，下载后用浏览器查看
# 实现效果：
![img](https://github.com/jjw-DL/YOLOV3-SORT/blob/master/output/output.gif) 

# 下载YOLOv3-416权重
- https://pjreddie.com/darknet/yolo/
- 基于Opencv的Dnn模块实现yolov3目标检测

# 卡尔曼滤波
- https://zhuanlan.zhihu.com/p/45238681
- Tutorial也有Kalman推导和基于python的实现
- 基于Filterpy实现Kalman滤波

# 指派问题与匈牙利解法
- 笔记：http://note.youdao.com/s/QsRIcLcJ
- 基于scipy.optimize的linear_sum_assignment实现匈牙利匹配
