"""Tracker 实现：门控最近邻（gated nearest neighbour）。

原理：
    逐帧往前推进，把当前帧的每个检测，匹配到上一帧某条轨迹的末端。
    只在"门控半径"内找最近的候选——超出半径就认为不是同一个雨滴，
    宁可断开轨迹，也不要跨越太远硬连。

为什么够用：
    本项目目标稀疏（实测一帧 8-17 个），雨滴基本沿重力方向直线下落，
    帧间位移远小于目标间距。这种场景下最近邻已经很稳，
    不需要上 Kalman。交接文档也明确：Kalman 只辅助预测和平滑，
    不能替代检测，检测质量决定跟踪上限。

性能：
    只处理稀疏的检测列表，不碰整幅图像，成本可以忽略（微秒级）。
    所以它留在批内串行执行，不影响并行度。

关于时间与速度（必须遵守的约束）：
    这个模块只输出"像素位移/帧"，不输出速度。
    逐帧真实时间戳尚未确认，禁止用 帧号差/180 换算成秒。
    速度必须等时间基准确认后，在 physics.py 里计算。

纯函数要求：
    track() 必须是纯函数——同样的输入永远给出同样的输出，
    包括 track_id 的编号顺序。否则并行跑和单线程跑的结果对不上，
    run_offline 的 --workers 1 vs --workers 8 回归测试会失败。
"""

from __future__ import annotations

import math

from ..registry import register_tracker
from ..types import Detection, Track


@register_tracker("nn_gate")
class NearestNeighbourGateTracker:
    """门控最近邻跟踪。

    参数：
        max_dist_px   : 门控半径（像素）。相邻帧位移超过它就不连。

                        默认值 150 的来由：实测本项目数据，雨滴帧间垂直位移
                        主峰在 20-140 px/帧（中位 83）。半径必须覆盖这个范围，
                        否则正常轨迹根本连不上——最初这里写的是 40，
                        结果 180 帧里只连上 15% 的检测，契约测试直接报红。

                        但要注意：圆形门控对下落雨滴本身就不合适。
                        半径开到能容纳 dy=140 时，它同时也放行了 dx=±140 的
                        横向跳变，会把旁边另一颗雨滴接进来。
                        实测 200px 时轨迹内 dy 标准差从 0.9 恶化到 1.4。
                        生产用途请选 gravity_gate（各向异性 + 单向门控）。
                        本实现保留作为对照基准。
        max_gap       : 允许轨迹中断几帧后再续上。0 表示必须逐帧连续。
                        设 1-2 可以容忍偶发漏检，但也更容易接错。
        min_track_len : 短于这个长度的轨迹丢弃。2 表示至少要出现在 2 帧里
                        （只有这样才能算位移）。
    """

    def __init__(
        self,
        max_dist_px: float = 150.0,
        max_gap: int = 0,
        min_track_len: int = 2,
    ):
        self.max_dist_px = float(max_dist_px)
        self.max_gap = int(max_gap)
        self.min_track_len = int(min_track_len)

    def track(self, per_frame: list[list[Detection]], cam: str) -> list[Track]:
        # active[i] = (Track, 上次更新时所在的列表下标, 末端检测)
        active: list[Track] = []
        finished: list[Track] = []
        # 每条活跃轨迹最后一次被更新时，对应 per_frame 的下标
        last_seen: dict[int, int] = {}

        next_id = 0

        for frame_pos, dets in enumerate(per_frame):
            # 1) 先把太久没更新的轨迹收掉
            still_active = []
            for tr in active:
                if frame_pos - last_seen[tr.track_id] > self.max_gap + 1:
                    finished.append(tr)
                else:
                    still_active.append(tr)
            active = still_active

            if not dets:
                continue

            # 2) 计算所有 (轨迹, 检测) 的距离，收集门控内的候选
            #    这里用朴素的两重循环。目标稀疏（个位数到几十），
            #    比建 KD 树还快，而且代码直白。
            candidates: list[tuple[float, int, int]] = []  # (距离, 轨迹下标, 检测下标)
            for ti, tr in enumerate(active):
                tail = tr.detections[-1]
                for di, det in enumerate(dets):
                    dist = math.hypot(det.cx - tail.cx, det.cy - tail.cy)
                    if dist <= self.max_dist_px:
                        candidates.append((dist, ti, di))

            # 3) 贪心分配：按距离从近到远，每条轨迹和每个检测各只用一次。
            #    排序键里带上 ti 和 di，保证距离相同时顺序也是确定的——
            #    这是"纯函数"要求的一部分。
            candidates.sort(key=lambda c: (c[0], c[1], c[2]))

            used_tracks: set[int] = set()
            used_dets: set[int] = set()

            for dist, ti, di in candidates:
                if ti in used_tracks or di in used_dets:
                    continue
                tr = active[ti]
                tr.detections.append(dets[di])
                last_seen[tr.track_id] = frame_pos
                used_tracks.add(ti)
                used_dets.add(di)

            # 4) 没被任何轨迹接走的检测，各自开一条新轨迹
            for di, det in enumerate(dets):
                if di in used_dets:
                    continue
                tr = Track(track_id=next_id, cam=cam, detections=[det])
                next_id += 1
                active.append(tr)
                last_seen[tr.track_id] = frame_pos

        finished.extend(active)

        # 5) 丢掉太短的轨迹，按 track_id 排序保证输出顺序确定
        result = [t for t in finished if t.n_frames >= self.min_track_len]
        result.sort(key=lambda t: t.track_id)
        return result
