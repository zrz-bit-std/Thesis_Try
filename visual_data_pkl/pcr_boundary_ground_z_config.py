import os


# bev-local, boundary contour of point cloud range
#         -----------top------------
#         |                        |
#         |                        |
#        left                     right
#         |                        |
#         |                        |
#         |                        |
#         ---------bottome----------

delta = 0.12  # car底盘到路面距离
# 首先从训练集中收集边界区域的车辆信息，估计地面高度！
pcr_boundary_ground_z = {
    "hy_7": {
        "intersection": {
            "top": -1.0405629513252208 - delta,
            "bottom": 1.090539356834054 - delta,
            "left": 0.7228434469790384 - delta,
            "right": -0.32669661049382714 - delta,
        },
        "roadsection": {
            "top": 0.5242372414307005 - delta,
            "bottom": -0.3757743132022472 - delta,
            "left": 0.2379120644307149 - delta,
            "right": -0.3447663573309921 - delta,
        }
    },
    "sh_2": {
        "intersection": {
            "top": -0.004152241573033707 - delta,
            "bottom": -0.24539024616695063 - delta,
            "left": -0.08187048204787234 - delta,
            "right": -0.05778761254501802 - delta,
        }
    },
    "sh_5": {
        "intersection": {
            "top": 0.3748477026329376 - delta,
            "bottom": 0 - delta,  # TODO: 使用新数据更新！！
            "left": 0.3996875 - delta,
            "right": 0.3647801147227535 - delta,
        }
    },
    "sh_6": {
        "intersection": {
            "top": 1.2826309545537506 - delta,
            "bottom": 0 - delta,   # TODO: 使用新数据更新！！
            "left": 1.3060419906687402 - delta,
            "right": 1.5945911647935138 - delta,
        }
    },
    "sh_3": {
        "intersection": {
            "top": -0.012661049019607854 - delta,
            "bottom": -0.06733928793103448 - delta,
            "left": -0.1035808623853211 - delta,
            "right": 0.19710866187050358 - delta,
        }
    },
    "sh_4": {
        "intersection": {
            "top": 0 - delta,  # TODO: 使用新数据更新！！
            "bottom": 0 - delta,  # TODO: 使用新数据更新！！
            "left": -0.12553515217391306 - delta,
            "right": -0.0011420262008733707 - delta,
        }
    },
}