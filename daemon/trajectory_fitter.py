import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter


# ========= 1) CV-Kalman + RTS 后向平滑 =========
class CVKalmanRTS:
    """
    Nearly-constant-velocity model in 2D with RTS smoothing.
    State: [x, y, vx, vy]
    Measurement: [x, y]
    Discrete Q(dt) from white-acc noise:
        [[dt^4/4,        0, dt^3/2,       0],
         [      0, dt^4/4,       0, dt^3/2],
         [dt^3/2,        0,   dt^2,       0],
         [      0, dt^3/2,       0,   dt^2]] * q_base
    """
    def __init__(self, meas_var=1e-2, q_base=1e-3, q_boost=3.0, a_cap=8.0):
        self.meas_var = float(meas_var)
        self.q_base   = float(q_base)
        self.q_boost  = float(q_boost)  # 自适应放大的上限比例
        self.a_cap    = float(a_cap)    # 加速度截断，避免过放大
        self.H = np.eye(2, 4)
        self.R = self.meas_var * np.eye(2)

    @staticmethod
    def _Q_cv(dt, q_base):
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        Q = np.array([[dt4/4,     0.0, dt3/2,    0.0],
                      [   0.0, dt4/4,    0.0, dt3/2],
                      [dt3/2,     0.0,   dt2,    0.0],
                      [   0.0,  dt3/2,    0.0,   dt2]], dtype=float)
        return q_base * Q

    def filter_and_smooth(self, measurements, timestamps, init_state=None):
        """ measurements: (N,2) ; timestamps: (N,)  """
        N = len(measurements)
        if N == 0:
            return np.zeros((0,4))

        z = np.asarray(measurements, dtype=float)
        t = np.asarray(timestamps, dtype=float)

        # 用真实时间，但平移到从 0 开始
        t = t - t[0]

        xs_f  = np.zeros((N, 4))
        Ps_f  = np.zeros((N, 4, 4))
        xs_p1 = np.zeros((N, 4))      # k+1 先验
        Ps_p1 = np.zeros((N, 4, 4))

        # init
        if init_state is None:
            x0 = np.zeros(4, dtype=float)
            x0[0:2] = z[0]
            if N >= 2:
                dt = max(1e-3, t[1] - t[0])
                x0[2:4] = (z[1] - z[0]) / dt
            P0 = np.diag([1, 1, 1, 1]).astype(float)
        else:
            x0, P0 = init_state

        x = x0.copy()
        P = P0.copy()

        # 基于观测的粗略加速度估计（自适应 Q）
        def est_acc(k):
            if k < 2:
                return 0.0
            dt1 = max(1e-6, t[k] - t[k-1])
            dt0 = max(1e-6, t[k-1] - t[k-2])
            a = (z[k] - 2*z[k-1] + z[k-2]) / (dt1*dt0)
            return float(np.linalg.norm(a))

        for k in range(N):
            if k == 0:
                xs_f[k] = x
                Ps_f[k] = P
                continue

            dt = max(1e-6, t[k] - t[k-1])
            F = np.eye(4)
            F[0,2] = dt
            F[1,3] = dt

            a_norm = est_acc(k)
            boost  = 1.0 + min(self.q_boost, a_norm / max(1e-6, self.a_cap))
            Q = self._Q_cv(dt, self.q_base * boost)

            # predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # update
            y  = z[k] - self.H @ x_pred
            S  = self.H @ P_pred @ self.H.T + self.R
            K  = P_pred @ self.H.T @ np.linalg.inv(S)
            x  = x_pred + K @ y
            P  = (np.eye(4) - K @ self.H) @ P_pred

            xs_f[k]  = x
            Ps_f[k]  = P
            xs_p1[k] = x_pred
            Ps_p1[k] = P_pred

        # RTS backward smoothing
        xs_s = xs_f.copy()
        Ps_s = Ps_f.copy()
        for k in range(N-2, -1, -1):
            dt = max(1e-6, t[k+1] - t[k])
            F = np.eye(4)
            F[0,2] = dt
            F[1,3] = dt

            Pk = Ps_f[k]
            Pk1_pred = Ps_p1[k+1]
            Ck = Pk @ F.T @ np.linalg.inv(Pk1_pred)

            xs_s[k] += Ck @ (xs_s[k+1] - xs_p1[k+1])
            Ps_s[k] += Ck @ (Ps_s[k+1] - Pk1_pred) @ Ck.T

        return xs_s  # (N,4)


# ========= 2) 工具函数 =========
def unwrap_angles(angles, reference=None):
    if reference is None:
        reference = angles[0]
    ang = np.unwrap(angles, discont=np.pi)
    offset = (reference - ang[0] + np.pi) % (2*np.pi) - np.pi
    return ang + offset

def wrap_angles(angles):
    return (angles + np.pi) % (2*np.pi) - np.pi

def savgol_safe(arr, window=7, poly=2):
    n = len(arr)
    if n < 3: return arr
    window = min(window, (n//2)*2 + 1)
    if window < 3: return arr
    try:
        return savgol_filter(arr, window_length=window, polyorder=min(poly, window-1))
    except Exception:
        return arr

def hermite_blend(p0, v0, p1, v1, tau):
    # 立方 Hermite：C¹ 连续过渡
    t = float(np.clip(tau, 0.0, 1.0))
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =      t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =      t**3 -   t**2
    return h00*p0 + h10*v0 + h01*p1 + h11*v1


# ========= 3) 主流程（样条用真实时间归一化 t_uniform） =========
class TrajectoryFitter:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.kalman_meas_var = 1e-2

        # 分段阈值（m/s）
        self.static_thr_up   = 0.20   # 进入动态
        self.static_thr_down = 0.12   # 回到静止（迟滞）
        self.min_seg_len     = 4

    @staticmethod
    def _is_valid(x, y):
        return not (np.any(np.isnan(x)) or np.any(np.isinf(x)) or
                    np.any(np.isnan(y)) or np.any(np.isinf(y)))

    def _speed_mps(self, xy, t):
        dt = np.diff(t)
        dt[dt <= 1e-9] = 1e-9
        v = np.linalg.norm(np.diff(xy, axis=0) / dt[:, None], axis=1)
        return np.append(v, v[-1] if len(v) else 0.0)

    def _segment_static_dynamic(self, boxes_np, timestamps, is_small=False, is_large=False):
        t = np.asarray(timestamps, dtype=float)
        xy = boxes_np[:, :2]
        v  = self._speed_mps(xy, t)

        # 目标大小微调阈值
        up, down = self.static_thr_up, self.static_thr_down
        if is_small:
            up *= 0.7; down *= 0.7
        elif is_large:
            up *= 1.2; down *= 1.2

        mask_dyn = np.zeros(len(v), dtype=bool)
        moving = False
        for i in range(len(v)):
            if not moving and v[i] >= up:    moving = True
            if moving and v[i] <= down:      moving = False
            mask_dyn[i] = moving

        # 生成段（保证最小长度）
        segments = []
        cur_state = mask_dyn[0]
        start = 0
        for i in range(1, len(mask_dyn)):
            if mask_dyn[i] != cur_state:
                if i - start < self.min_seg_len:
                    continue  # 段太短则并入
                segments.append((start, i, cur_state))
                start = i
                cur_state = mask_dyn[i]
        segments.append((start, len(mask_dyn), cur_state))
        return segments  # (s, e, is_dynamic)

    def _fit_dynamic_segment(self, seg_boxes, seg_t, is_small, is_large):
        t = np.asarray(seg_t, dtype=float)
        t_min, t_max = t[0], t[-1]
        if t_max > t_min:
            t_uniform = (t - t_min) / (t_max - t_min)
        else:
            t_uniform = np.zeros_like(t)

        x = seg_boxes[:, 0].astype(float)
        y = seg_boxes[:, 1].astype(float)

        # 样条平滑强度（根据目标大小 & 点数）
        if is_small:
            s_base = 0.02
            q_base = 6e-4
        elif is_large:
            s_base = 0.01
            q_base = 2e-3   # 大目标过程噪声大些，减滞后
        else:
            s_base = 0.012
            q_base = 1.2e-3

        s = s_base * np.log(len(t_uniform) + 1.0)
        k = min(3, max(1, len(t_uniform) - 1))

        # === 关键：用真实时间归一化后的 t_uniform 做 u ===
        if len(t_uniform) <= 1:
            return seg_boxes, np.zeros((len(seg_boxes), 2))
        tck, _ = splprep([x, y], u=t_uniform, s=s, k=k)
        x_fit, y_fit = splev(t_uniform, tck)  # 直接在原采样位置求值

        fitted = np.copy(seg_boxes)
        fitted[:, 0] = x_fit
        fitted[:, 1] = y_fit

        # KF + RTS：自适应 Q 的 CV 模型（用真实时间 t）
        kf = CVKalmanRTS(meas_var=self.kalman_meas_var, q_base=q_base, q_boost=3.0, a_cap=8.0)
        xs = kf.filter_and_smooth(fitted[:, :2], t)  # (N,4)
        fitted[:, 0] = xs[:, 0]
        fitted[:, 1] = xs[:, 1]

        return fitted, xs[:, 2:4]  # 返回位置和速度

    # def _smooth_yaw_dynamic(self, boxes_np, use_path_yaw=False):
    #     if use_path_yaw:
    #         # 用轨迹切线角 + 轻量 SG
    #         x, y = boxes_np[:, 0], boxes_np[:, 1]
    #         dx = np.gradient(x)
    #         dy = np.gradient(y)
    #         path_yaw = np.arctan2(dy, dx)
    #         yaw = unwrap_angles(path_yaw)
    #     else:
    #         yaw = unwrap_angles(boxes_np[:, 6])

    #     yaw = savgol_safe(yaw, window=7, poly=2)
    #     boxes_np[:, 6] = wrap_angles(yaw)
    #     return boxes_np

    def fit_with_time(self, boxes_np, timestamps, is_small=False, is_large=False, use_path_yaw=False):
        if len(boxes_np) < 3 or len(timestamps) != len(boxes_np):
            return boxes_np, False

        x, y = boxes_np[:, 0], boxes_np[:, 1]
        if not self._is_valid(x, y):
            return boxes_np, False

        segments = self._segment_static_dynamic(boxes_np, timestamps, is_small, is_large)

        fitted = np.copy(boxes_np)
        vel_store = np.zeros((len(boxes_np), 2))  # 保存端点速度，用于段间过渡

        for (s, e, is_dyn) in segments:
            seg = boxes_np[s:e]
            seg_t = np.asarray(timestamps[s:e], dtype=float)

            if not is_dyn:
                # 静止段：位置均值，yaw 中位数
                mean_box = seg[0].copy()
                mean_box[0] = np.mean(seg[:, 0])
                mean_box[1] = np.mean(seg[:, 1])
                mean_box[6] = np.median(seg[:, 6])
                fitted[s:e] = mean_box
                vel_store[s:e] = 0.0
            else:
                seg_fit, seg_vel = self._fit_dynamic_segment(seg, seg_t, is_small, is_large)
                seg_fit = self._smooth_yaw_dynamic(seg_fit, use_path_yaw=use_path_yaw)
                fitted[s:e] = seg_fit
                vel_store[s:e] = seg_vel

        # 段间 C¹ 过渡（Hermite）
        t = np.asarray(timestamps, dtype=float)
        L = 5  # 过渡窗口帧数
        for i in range(1, len(segments)):
            s0, e0, dyn0 = segments[i-1]
            s1, e1, dyn1 = segments[i]
            k0 = e0 - 1
            k1 = s1
            if k0 < 1 or k1 >= len(fitted):
                continue

            p0 = fitted[k0, :2]
            p1 = fitted[k1, :2]
            v0 = vel_store[k0, :2]
            v1 = vel_store[k1, :2]

            # 用平均 dt 设定 Hermite 尺度
            if len(t) > 1:
                dt_mean = np.median(np.diff(t))
            else:
                dt_mean = 1.0
            T = max(dt_mean * L, 1e-3)
            m0 = v0 * T
            m1 = v1 * T

            # 仅平滑下一段的前 L 帧
            for j in range(L):
                idx = k1 + j
                if idx >= len(fitted): break
                tau = (j + 1) / (L + 1)
                fitted[idx, :2] = hermite_blend(p0, m0, p1, m1, tau)

        return fitted, True

    @staticmethod
    def _wrap_to_pi(a):
        # 映射到 [-pi, pi)
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _angle_diff(a, b):
        # 返回 a - b 的最小符号差（在 [-pi, pi)）
        return TrajectoryFitter._wrap_to_pi(a - b)

    @staticmethod
    def _unwrap_continuous(angles):
        """对一串角度做连续展开，避免跨 ±pi 的跳变。"""
        if len(angles) == 0:
            return angles
        unwrapped = np.array(angles, dtype=float)
        for i in range(1, len(unwrapped)):
            diff = unwrapped[i] - unwrapped[i-1]
            diff = TrajectoryFitter._wrap_to_pi(diff)
            unwrapped[i] = unwrapped[i-1] + diff
        return unwrapped

    @staticmethod
    def _savgol_safe_1d(sig, window=7, poly=2):
        n = len(sig)
        if n == 0:
            return sig
        w = min(window, n if n % 2 == 1 else n - 1)
        if w < 3 or w <= poly:
            return sig if n < 3 else np.convolve(sig, np.ones(min(3, n))/min(3, n), mode='same')
        return savgol_filter(sig, w, poly, mode='interp')

    def safe_gradient(data, edge_order=1):
        if len(data) < edge_order + 1:
            # 返回零梯度或适当的值
            return np.zeros_like(data)
        return np.gradient(data, edge_order=edge_order)

    def _smooth_yaw_dynamic(self, boxes_np, use_path_yaw=False, min_speed=1e-3,
                            sg_window=7, sg_poly=2):
        boxes_np = np.asarray(boxes_np, dtype=float)
        T = boxes_np.shape[0]
        if T == 0:
            return boxes_np

        x, y = boxes_np[:, 0], boxes_np[:, 1]

        if len(boxes_np) < 2:
            dx = np.zeros_like(x)  # 可按需改成基于 timestamps 的 dx/dt
            dy = np.zeros_like(y)
        else:
            dx = np.gradient(x)  # 可按需改成基于 timestamps 的 dx/dt
            dy = np.gradient(y)
        # dx = np.gradient(x)  # 可按需改成基于 timestamps 的 dx/dt
        # dy = np.gradient(y)
        speed = np.hypot(dx, dy)
        path_yaw = np.arctan2(dy, dx)

        yaw_raw = path_yaw.copy() if use_path_yaw else boxes_np[:, 6].copy()

        yaw_aligned = yaw_raw.copy()
        moving = speed > min_speed
        if moving.any():
            cls = self.__class__
            diff = cls._angle_diff(yaw_aligned[moving], path_yaw[moving])

            flips = np.abs(diff) > (np.pi / 2)
            # ---- 修复：避免链式布尔索引导致不回写 ----
            idx = np.where(moving)[0]
            sub = idx[flips]
            if len(sub) > 0:
                yaw_aligned[sub] = cls._wrap_to_pi(yaw_aligned[sub] + np.pi)

        cls = self.__class__
        yaw_unwrapped = cls._unwrap_continuous(yaw_aligned)
        yaw_smooth = cls._savgol_safe_1d(yaw_unwrapped, window=sg_window, poly=sg_poly)
        yaw_out = cls._wrap_to_pi(yaw_smooth)

        boxes_np[:, 6] = yaw_out
        return boxes_np


