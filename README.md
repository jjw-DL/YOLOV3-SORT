# YOLOV3-SORT实现车辆跟踪与车流统计
- 基于Opencv的Dnn模块实现yolov3进行目标检测
- 基于Filterpy实现Kalman滤波
- 基于scipy.optimize的linear_sum_assignment实现匈牙利匹配
# 实现效果：
![img](https://github.com/jjw-DL/YOLOV3-SORT/blob/master/output/output.gif) 

# 下载YOLOv3-416权重
- https://pjreddie.com/darknet/yolo/

- 代码简单，适合新手学习，ID Switch较多。
- 先学习SORT算法，明白基于检测的目标跟踪基本框架，然后在学习Deepsort。

# 指派问题与匈牙利解法优秀博客推荐：
- https://www.cnblogs.com/chenyg32/p/3293247.html
- https://blog.csdn.net/Wonz5130/article/details/80678410
