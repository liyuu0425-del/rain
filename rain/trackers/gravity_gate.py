"""Tracker 实现：重力门控（各向异性 + 单向）。

## 为什么需要这个实现

圆形门控（nn_gate）用一个半径同时管住 x 和 y 方向，对下落的雨滴是错的。

实测本项目数据（cama/Capture_20260727_232420，180 帧）：
    限定 |dx| < 15 px 的候选配对中，dy 的分布是
        p10=+36  p25=+50  中位=+83  p75=+106  p90=+129
    直方图主峰在 20-140 px/帧，140 以上迅速衰减为噪声。
    dx 中位数 +1，完全对称 —— 确认雨滴基本垂直下落。

所以圆形门控 40 px 的问题是双向的：
    - 太小：挡掉了正常的 ~83 px 垂直位移，只有 12% 的配对能进门
      （实测 180 帧只连出 38 条轨迹）
    - 放大到 200 px 也不对：它同时放行了 dx = ±200 的横向跳变，
      会把旁边另一颗雨滴接进来。实测轨迹数在 200 px 达峰后开始下降，
      正是轨迹被错误合并的信号。

正确做法是按物理分开约束：
    y 方向：只允许向下，范围宽（dy_min .. dy_max）
    x 方向：允许小幅横移（风、摆动、质心噪声），范围窄（|dx| <= dx_max）

## 参数当前状态

下面的默认值来自上述实测分布，**是这一批数据的经验值，不是最终参数**。
必须用人工标注验证集标定，并在不同雨强下复核。
交接文档明确：不要直接采用示例参数作为最终参数。

## 时间约束

本模块只输出像素位移/帧，不输出速度。
dy_min/dy_max 的物理含义要等像素-毫米标定和逐帧时间戳确认后才能解释。
"""

from __future__ import annotations

from ..registry import register_tracker
from ..types import Detection, Track


@register_tracker("gravity_gate")
class GravityGateTracker:
    """各向异性单向门控最近邻跟踪。

    参数：
        dy_min, dy_max : 帧间垂直位移的允许区间（像素，向下为正）。
                         实测主峰 20-140，默认取 15..170 留些余量。
                         dy_min > 0 表示强制向下 —— 雨滴不会向上飞，
                         这一条约束就能排掉大量错误配对。
        dx_max         : 帧间水平位移绝对值上限（像素）。

                         默认 20 来自实测的召回/误连权衡（90 帧，724 个检测）：
                             dx_max=45: 轨迹69 用掉94% 倾斜>8.5°的16% dy标准差0.9
                             dx_max=30: 轨迹67 用掉92% 倾斜13%        dy标准差0.7
                             dx_max=20: 轨迹64 用掉87% 倾斜 8%        dy标准差0.6  <- 默认
                             dx_max=12: 轨迹61 用掉80% 倾斜 5%        dy标准差0.6
                             dx_max= 8: 轨迹55 用掉75% 倾斜 0%        dy标准差0.6
                         放太松会让轨迹斜着穿过画面（把旁边另一颗雨滴接进来），
                         收太紧会丢掉真实轨迹。这个权衡点必须用人工标注确定，
                         当前默认值只是个合理起点。
        cost_dx_weight : 代价函数里 x 方向偏移的权重。
                         设 >1 表示"横向偏移比纵向更可疑"，
                         因为纵向位移本来就该有，横向不该有。
        max_gap        : 允许中断几帧后续上。0 = 必须逐帧连续。
        min_track_len  : 短于此长度的轨迹丢弃。2 = 至少出现在 2 帧里。
    """

    def __init__(
        self,
        dy_min: float = 15.0,
        dy_max: float = 170.0,
        dx_max: float = 20.0,
        cost_dx_weight: float = 3.0,
        max_gap: int = 0,
        min_track_len: int = 2,
    ):
        self.dy_min = float(dy_min)
        self.dy_max = float(dy_max)
        self.dx_max = float(dx_max)
        self.cost_dx_weight = float(cost_dx_weight)
        self.max_gap = int(max_gap)
        self.min_track_len = int(min_track_len)

    def _predict(self, track: Track) -> tuple[float, float]:
        """预测下一帧位置。

        有两帧以上就用最近一次的位移做线性外推——雨滴接近匀速，
        外推能显著收窄搜索范围。只有一帧时退回用当前位置。

        注意：这是纯几何外推，不是 Kalman。交接文档明确要求
        预测不能替代检测，这里预测只用来算代价、缩小门控。
        """
        tail = track.detections[-1]
        if len(track.detections) < 2:
            return tail.cx, tail.cy
        prev = track.detections[-2]
        dn = tail.frame_idx - prev.frame_idx
        if dn <= 0:
            return tail.cx, tail.cy
        vx = (tail.cx - prev.cx) / dn
        vy = (tail.cy - prev.cy) / dn
        return tail.cx + vx, tail.cy + vy

    def track(self, per_frame: list[list[Detection]], cam: str) -> list[Track]:
        active: list[Track] = []
        finished: list[Track] = []
        last_seen: dict[int, int] = {}
        next_id = 0

        for frame_pos, dets in enumerate(per_frame):
            # 1) 收掉太久没更新的轨迹
            still_active = []
            for tr in active:
                if frame_pos - last_seen[tr.track_id] > self.max_gap + 1:
                    finished.append(tr)
                else:
                    still_active.append(tr)
            active = still_active

            if not dets:
                continue

            # 2) 收集门控内的候选。目标稀疏（每帧 2-14 个），朴素两重循环最快。
            candidates: list[tuple[float, int, int]] = []
            for ti, tr in enumerate(active):
                tail = tr.detections[-1]
                px, py = self._predict(tr)

                for di, det in enumerate(dets):
                    # --- 门控：按物理分开约束 x 和 y ---
                    dy = det.cy - tail.cy      # 相对实际末端位置判断运动方向
                    dx = det.cx - tail.cx

                    if dy < self.dy_min or dy > self.dy_max:
                        continue                # 不向下、或掉得太远
                    if abs(dx) > self.dx_max:
                        continue                # 横向跳变太大

                    # --- 代价：相对预测位置的偏差，x 方向加权 ---
                    ex = abs(det.cx - px) * self.cost_dx_weight
                    ey = abs(det.cy - py)
                    cost = ex + ey
                    candidates.append((cost, ti, di))

            # 3) 贪心分配。排序键带上 ti/di，保证代价相同时顺序也确定 ——
            #    这是"纯函数"要求的一部分，否则并行与单线程结果对不上。
            candidates.sort(key=lambda c: (c[0], c[1], c[2]))

            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            for cost, ti, di in candidates:
                if ti in used_tracks or di in used_dets:
                    continue
                tr = active[ti]
                tr.detections.append(dets[di])
                last_seen[tr.track_id] = frame_pos
                used_tracks.add(ti)
                used_dets.add(di)

            # 4) 没被接走的检测各开一条新轨迹
            for di, det in enumerate(dets):
                if di in used_dets:
                    continue
                tr = Track(track_id=next_id, cam=cam, detections=[det])
                next_id += 1
                active.append(tr)
                last_seen[tr.track_id] = frame_pos

        finished.extend(active)

        result = [t for t in finished if t.n_frames >= self.min_track_len]
        result.sort(key=lambda t: t.track_id)
        return result
