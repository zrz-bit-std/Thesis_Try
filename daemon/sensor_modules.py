# default modules setting
intersection_modules = [
    "camera_0_0",
    "camera_1_0",
    "camera_2_0",
    "camera_3_0",
    "camera_0_8",
    "camera_1_8",
    "camera_2_8",
    "camera_3_8",
    "lidar_0_12",
    "lidar_1_12",
    "lidar_2_12",
    "lidar_3_12",
]

intersection_lidar_modules = [
    "lidar_0_12",
    "lidar_1_12",
    "lidar_2_12",
    "lidar_3_12",
]

roadsection_modules = [
    "camera_0_1", 
    "camera_1_1", 
    "camera_2_1", 
    "camera_3_1",
    "lidar_0_11",
    "lidar_1_11",
    "lidar_2_11",
    "lidar_3_11",
]

single_pole_modules = {
    "S0": ["camera_0_1", "lidar_0_11"],
    "S1": ["camera_1_1", "lidar_1_11"],
    "S2": ["camera_2_1", "lidar_2_11"],
    "S3": ["camera_3_1", "lidar_3_11"],
}
sensors2poles = {}
for cur_pole, cur_sensors in single_pole_modules:
    for cur_sensor in cur_sensors:
        sensors2poles[cur_sensor] = cur_pole

roadside_sensor_modules = {
    "hy_7": {
        "intersection": intersection_modules,
        "roadsection": roadsection_modules,
    },
    "sh_2": {
        "intersection": intersection_modules,
        "roadsection": roadsection_modules,
    },
    "sh_3": {
        "intersection": intersection_modules,
    },
    "sh_4": {  # 丁字路口
        "intersection": [
            "camera_0_0",
            "camera_1_0",
            "camera_3_0",
            "camera_0_8",
            "camera_1_8",
            "camera_3_8",
            "lidar_0_12",
            "lidar_1_12",
            "lidar_3_12",
        ],
    },
    "tx_1": {
        "intersection": intersection_modules,
    },
    "hy_10": {
        "intersection": [
            "camera_0_0",
            "camera_1_1",  # 10号S1杆位顺向/逆向槽位命名反了
            "camera_2_0",
            "camera_3_0",
            "camera_0_8",
            "camera_1_8",
            "camera_2_8",
            "camera_3_8",
            "lidar_0_12",
            "lidar_1_11",  # 10号S1杆位顺向/逆向槽位命名反了
            "lidar_2_12",
            "lidar_3_12",
        ]
    },
    "sh_5": {
        "intersection": [
            "camera_0_1",  # S0 camera顺向/逆向槽位命名反了
            "camera_1_0",  
            "camera_2_0",
            "camera_3_1",  # S3 camera顺向/逆向槽位命名反了
            "camera_0_8",
            "camera_1_8",
            "camera_2_8",
            "camera_3_8",
            "lidar_0_12",
            "lidar_1_12",
            "lidar_2_12",
            "lidar_3_12",
        ]
    },
    "sh_6": {
        "intersection": ["camera_0_0", 
                         "camera_1_0", 
                         "camera_3_0",
                         "camera_0_8", 
                         "camera_1_8", 
                         "camera_3_8",
                         "lidar_0_12",  
                         "lidar_1_12", 
                         "lidar_3_12",],
        "roadsection": ["camera_0_1", 
                        "camera_1_1", 
                        "camera_3_1",
                        "lidar_0_11"],
    },
}

# check, 避免书写重复错误
for _, road_modules_info in roadside_sensor_modules.items():
    for road_type, modules in road_modules_info.items():
        assert len(modules) == len(set(modules))


single_pole_roadside_sensor_modules = {
    "hy_7": {
        "roadsection": single_pole_modules,
    },
}



# 同一朝向camera名字和lidar名字映射表
camera_lidar_pairs = {
    "camera_0_0": "lidar_0_12",
    "camera_0_1": "lidar_0_11",
    "camera_1_0": "lidar_1_12",
    "camera_1_1": "lidar_1_11",
    "camera_2_0": "lidar_2_12",
    "camera_2_1": "lidar_2_11",
    "camera_3_0": "lidar_3_12",
    "camera_3_1": "lidar_3_11",
}
lidar_camera_pairs = {
    lidar_name: camera_name
    for camera_name, lidar_name in camera_lidar_pairs.items()
}

road_lidar_camera_pairs = {
    "hy_7": lidar_camera_pairs,
    "hy_10": lidar_camera_pairs,
    "sh_2": lidar_camera_pairs,
    "sh_3": lidar_camera_pairs,
    "sh_4": lidar_camera_pairs,
    "sh_5": {
        # note_20250606: sh_5 暂时没有路段激光数据 
        # "lidar_0_11": "camera_0_0", # s0 反了
        # "lidar_1_11": "camera_1_1", 
        # "lidar_2_11": "camera_2_1",
        # "lidar_3_11": "camera_3_0", # s3 反了
        "lidar_0_12": "camera_0_1", # s0 反了
        "lidar_1_12": "camera_1_0",
        "lidar_2_12": "camera_2_0",
        "lidar_3_12": "camera_3_1", # s3 反了
    },
    "sh_6": {"lidar_0_11": "camera_0_1",
             "lidar_0_12": "camera_0_0",
             "lidar_1_12": "camera_1_0",
             "lidar_3_12": "camera_3_0",},
    "tx_1": lidar_camera_pairs,
}

road_camera_lidar_pairs = {
    "sh_2": camera_lidar_pairs,
    "sh_3": camera_lidar_pairs,
    "sh_4": camera_lidar_pairs,
    "sh_6": camera_lidar_pairs,
}
