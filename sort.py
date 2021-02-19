from __future__ import print_function
from numba import jit
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

@jit
def iou(bb_test, bb_gt):
    """
    在两个box间计算IOU
    :param bb_test: box1 = [x1y1x2y2]
    :param bb_gt: box2 = [x1y1x2y2]
    :return: 交并比IOU
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (
            bb_gt[3] - bb_gt[1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    将[x1,y1,x2,y2]形式的检测框转为滤波器的状态表示形式[x,y,s,r]。其中x，y是框的中心坐标，s是面积，尺度，r是宽高比
    :param bbox: [x1,y1,x2,y2] 分别是左上角坐标和右下角坐标
    :return: [ x, y, s, r ] 4行1列，其中x,y是box中心位置的坐标，s是面积，r是纵横比w/h
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    将[cx，cy，s，r]的目标框表示转为[x_min，y_min，x_max，y_max]的形式
    :param x:[ x, y, s, r ],其中x,y是box中心位置的坐标，s是面积，r
    :param score: 置信度
    :return:[x1,y1,x2,y2],左上角坐标和右下角坐标
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        """
        初始化边界框和跟踪器
        :param bbox:
        """
        # 定义等速模型
        # 内部使用KalmanFilter，7个状态变量和4个观测输入
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0  # 记录从上次更新到当前帧的预测次数，每次更新后清0(update函数中)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0  # 记录跟踪上的次数，一旦一帧没有跟上直接清0(predict函数中)
        self.age = 0

    def update(self, bbox):
        """
        使用观察到的目标框更新状态向量。filterpy.kalman.KalmanFilter.update 会根据观测修改内部状态估计self.kf.x。
        重置self.time_since_update，清空self.history。
        :param bbox:目标框
        :return:
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        推进状态向量并返回预测的边界框估计。
        将预测结果追加到self.history。由于 get_state 直接访问 self.kf.x，所以self.history没有用到
        :return:
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        # 预测次数
        self.age += 1
        # 若跟踪过程中未进行更新，将hit_streak = 0
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        # 将预测结果追加到history
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        返回当前边界框估计值
        :return:
        """
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    将检测框bbox与卡尔曼滤波器的跟踪框进行关联匹配
    :param detections:检测框
    :param trackers:跟踪框，即跟踪目标
    :param iou_threshold:IOU阈值
    :return:跟踪成功目标的矩阵：matchs
            新增目标的矩阵：unmatched_detections
            跟踪失败即离开画面的目标矩阵：unmatched_trackers
    """
    # 跟踪目标数量为0，直接构造结果
    if (len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # iou 不支持数组计算。逐个计算两两间的交并比，调用 linear_assignment 进行匹配
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    # 遍历目标检测的bbox集合，每个检测框的标识为d
    for d, det in enumerate(detections):
        # 遍历跟踪框（卡尔曼滤波器预测）bbox集合，每个跟踪框标识为t
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    # 通过匈牙利算法将跟踪框和检测框以[[d,t]...]的二维矩阵的形式存储在match_indices中
    result = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*result)))

    # 记录未匹配的检测框及跟踪框
    # 未匹配的检测框放入unmatched_detections中，表示有新的目标进入画面，要新增跟踪器跟踪目标
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    # 未匹配的跟踪框放入unmatched_trackers中，表示目标离开之前的画面，应删除对应的跟踪器
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    # 将匹配成功的跟踪框放入matches中
    matches = []
    for m in matched_indices:
        # 过滤掉IOU低的匹配，将其放入到unmatched_detections和unmatched_trackers
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        # 满足条件的以[[d,t]...]的形式放入matches中
        else:
            matches.append(m.reshape(1, 2))
    # 初始化matches,以np.array的形式返回
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
        初始化：设置SORT算法的关键参数       
        """
        # 最大检测数：目标未被检测到的帧数，超过之后会被删 
        self.max_age = max_age
        # 目标命中的最小次数，小于该次数不返回
        self.min_hits = min_hits  
        # 卡尔曼跟踪器
        self.trackers = []  
        # 帧计数
        self.frame_count = 0
    
    def update(self, dets):
        self.frame_count += 1
        # 在当前帧逐个预测轨迹位置，记录状态异常的跟踪器索引
        # 根据当前所有的卡尔曼跟踪器个数（即上一帧中跟踪的目标个数）创建二维数组：行号为卡尔曼滤波器的标识索引，列向量为跟踪框的位置和ID
        trks = np.zeros((len(self.trackers), 5))  # 存储跟踪器的预测
        to_del = []   # 存储要删除的目标框
        ret = []    # 存储要返回的追踪目标框
        # 循环遍历卡尔曼跟踪器列表
        for t, trk in enumerate(trks):
            # 使用卡尔曼跟踪器t产生对应目标的跟踪框
            pos = self.trackers[t].predict()[0]
            # 遍历完成后，trk中存储了上一帧中跟踪的目标的预测跟踪框
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # 如果跟踪框中包含空值则将该跟踪框添加到要删除的列表中
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # numpy.ma.masked_invalid 屏蔽出现无效值的数组（NaN 或 inf）
        # numpy.ma.compress_rows 压缩包含掩码值的2-D 数组的整行，将包含掩码值的整行去除
        # trks中存储了上一帧中跟踪的目标并且在当前帧中的预测跟踪框
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # 逆向删除异常的跟踪器，防止破坏索引
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 将目标检测框与卡尔曼滤波器预测的跟踪框关联获取跟踪成功的目标，新增的目标，离开画面的目标
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # 将跟踪成功的目标框更新到对应的卡尔曼滤波器
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                # 使用观测的边界框更新状态向量
                trk.update(dets[d, :][0])

        # 为新增的目标创建新的卡尔曼滤波器对象进行跟踪
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 自后向前遍历，仅返回在当前帧出现且命中周期大于self.min_hits（除非跟踪刚开始）的跟踪结果；如果未命中时间大于self.max_age则删除跟踪器。
        # hit_streak忽略目标初始的若干帧
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 返回当前边界框的估计值
            d = trk.get_state()[0]
            # 跟踪成功目标的box与id放入ret列表中
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # 跟踪失败或离开画面的目标从卡尔曼跟踪器中删除
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        # 返回当前画面中所有目标的box与id,以二维矩阵形式返回
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

