import cv2
import os
import numpy as np
import glob
import yaml


# 自定义解析opencv-matrix标签
def opencv_matrix_constructor(loader, node):
    # 解析opencv-matrix的data字段
    rows = int(node.value[0][1].value)  # rows值是一个ScalarNode，所以使用node.value直接访问
    cols = int(node.value[1][1].value)  # cols值是一个ScalarNode
    data = node.value[3][1].value  # data字段是一个SequenceNode，进一步获取数据
    
    # 将数据提取出来（node.value[3].value是一个包含数字的列表）
    data_values = [float(val.value) for val in data]
    
    # 将data转换为NumPy数组并按照rows, cols重塑
    # return np.array(data_values).reshape((rows, cols))
    return data_values
    # return None

# 注册自定义解析器
yaml.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)

# 读取 YAML 文件
def read_yaml_file(file_path):
    # 打开文件并读取所有行
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 去掉前两行和第49至50行
    # cleaned_lines = lines[2:48] + lines[50:]
    cleaned_lines = lines[2:]

    # 将处理后的内容合并为字符串
    cleaned_content = ''.join(cleaned_lines)

    # 使用 yaml.safe_load 解析剩余的内容
    try:
        data = yaml.load(cleaned_content, Loader=yaml.FullLoader)
        return data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    

def read_yaml(yaml_file, old_calib_version=False, intrinsic_yaml_file=None, resize_calib = True):
    # 新版标定文件
    if not old_calib_version:
        yaml_data = read_yaml_file(yaml_file)
        
        intrinsic = yaml_data['projection_parameters']

        T_common_cam = yaml_data['T_common_cam'][:12]
        h = yaml_data['image_height']
        w = yaml_data['image_width']
        # if h==w:
        #     mtx_cam = np.array([  # 内参矩阵
        #         [273.2846130956868, 0, 514.2933787741496],
        #         [0, 272.8518077656246, 519.3761308757942],
        #         [0, 0, 1]
        #     ])
        #     return mtx_cam, T_common_cam
        if yaml_data['model_type']=='fisheye-opencv':
            mtx_cam = np.array([  # 内参矩阵
                [intrinsic['mu'], 0, intrinsic['u0']],
                [0, intrinsic['mv'], intrinsic['v0']],
                [0, 0, 1]
            ])
            return mtx_cam, T_common_cam
        
        fx = intrinsic['mu']
        fy = intrinsic['mv']
        cx = intrinsic['u0']
        cy = intrinsic['v0']
        k1 = intrinsic['k1']
        k2 = intrinsic['k2']
        p1 = intrinsic['p1']
        p2 = intrinsic['p2']
        mtx_cam = np.array([  # 内参矩阵
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        dist_cam = np.array([  # 畸变系数
            k1, k2, p1, p2
        ])

        if h==2160 and w==3840:
            h = int(h/3)
            w = int(w/3)
            mtx_cam = mtx_cam/3
            mtx_cam[2][2]*=3

        return mtx_cam, dist_cam, h, w, T_common_cam
    # 老版标定文件
    else:
        yaml_data = read_yaml_file(yaml_file)
        intrinsic_yaml_data = read_yaml_file(intrinsic_yaml_file)
        
        intrinsic = intrinsic_yaml_data['projection_parameters']
        h = intrinsic_yaml_data['image_height']
        w = intrinsic_yaml_data['image_width']
        
        T_common_cam = yaml_data['T_common_cam'][:12]

        # if h==w:
        #     mtx_cam = np.array([  # 内参矩阵
        #         [273.2846130956868, 0, 514.2933787741496],
        #         [0, 272.8518077656246, 519.3761308757942],
        #         [0, 0, 1]
        #     ])
        #     return mtx_cam, T_common_cam
        if yaml_data['model_type']=='fisheye-opencv':
            ratio = 1024/yaml_data['dist_image_width']
                
            mtx_cam = np.array([  # 内参矩阵
                [intrinsic['mu']*ratio, 0, intrinsic['u0']*ratio],
                [0, intrinsic['mv']*ratio, intrinsic['v0']*ratio],
                [0, 0, 1]
            ])
            return mtx_cam, T_common_cam
        
        fx = intrinsic['mu']
        fy = intrinsic['mv']
        cx = intrinsic['u0']
        cy = intrinsic['v0']

        T_common_cam = yaml_data['T_common_cam'][:12]
        intrinsic_matrix = np.array([  # 内参矩阵
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        if h==2160 and w==3840:
            h = int(h/3)
            w = int(w/3)
            intrinsic_matrix = intrinsic_matrix/3
            intrinsic_matrix[2][2]*=3

        return intrinsic_matrix, T_common_cam


def undistort(img_path, yaml_file, new_img_path):
    mtx_cam, dist_cam, h, w, Tr = read_yaml(yaml_file)
    print('Tr', Tr)
    print('mtx_cam', mtx_cam)
    print('dist_cam', dist_cam)
    for file in os.listdir(img_path):
        img_file = os.path.join(img_path, file)
        img = cv2.imread(img_file)
        h1, w1 = img.shape[:2]
        if int(h1) == h and int(w1) == w:
            break
        else:
            if not os.path.exists(new_img_path):
                os.makedirs(new_img_path)
            img = cv2.resize(img, (w, h))
            # new_img_path = img_path + '_dis'
            # if not os.path.exists(new_img_path):
            #     os.makedirs(new_img_path)

            cv2.imwrite(os.path.join(new_img_path, file), img)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_cam, dist_cam, (w, h), 1, (w, h))

    return newcameramtx, Tr

def get_bev_param(yaml_file, cam_num, old_calib_version=True, intrinsic_yaml_file=None):
    if old_calib_version:
        intinsic_matrix, Tr = read_yaml(yaml_file, old_calib_version=True, intrinsic_yaml_file=intrinsic_yaml_file)

        Tr = np.array(Tr).reshape(3, 4)

        P0 = np.concatenate((intinsic_matrix, np.zeros((3, 1))), axis=1)
        P0 = P0.reshape(12).tolist()
        Rt = np.array(Tr[0:3, 0:3]).reshape(3, 3)
        T = np.array(Tr[0:3, 3]).reshape(3, 1)
        Rt_t_matrix = np.concatenate((Rt, T), axis=1)
        Tr_velo_to_cam = Rt_t_matrix.reshape(12).tolist()

        line1 = '%s' % (str(P0).replace(',', '')[1:-1])
        line2 = '%s' % (str(Tr_velo_to_cam).replace(',', '')[1:-1])
        print(f'P{cam_num}:', line1)
        print(f'Tr_velo_to_cam_{cam_num}:', line2)

        res = []
        res.append(f'P{cam_num}: {line1}')
        res.append(f'Tr_velo_to_cam_{cam_num}: {line2}')
        return res

        # with open(calib_file, 'w') as f:
        #     f.write('P0: %s\n' % (str(P0).replace(',', ' ')[1:-1]))
        #     f.write('Tr_velo_to_cam: %s\n' % (str(Tr_velo_to_cam).replace(',', ' ')[1:-1]))
        # print('*******save to %s*********' % calib_file)


def inverse_rigid_trans(Tr):
    inv_Tr = np.zeros_like(Tr)
    print(inv_Tr)
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def get_param(img_path, yaml_file, new_path, calib_file):
    Newcameramtx, Tr = undistort(img_path, yaml_file, new_path)
    Tr = np.array(Tr).reshape(3, 4)

    P0 = np.concatenate((Newcameramtx, np.zeros((3, 1))), axis=1)
    P0 = P0.reshape(12).tolist()
    print('P0: ', P0)
    Rt = np.array(Tr[0:3, 0:3]).reshape(3, 3)
    T = np.array(Tr[0:3, 3]).reshape(3, 1)
    print(T)
    Rt_t_matrix = np.concatenate((Rt, T), axis=1)
    # inv_Tr = inverse_rigid_trans(Rt_t_matrix)
    # Tr_velo_to_cam = inv_Tr.reshape(12).tolist()
    Tr_velo_to_cam = Rt_t_matrix.reshape(12).tolist()
    # R2 = Tr_velo_to_cam + [0, 0, 0, 1]
    #
    # R2 = np.array(R2).reshape(4, 4)
    # Tr_velo_to_cam = np.linalg.inv(R2)
    print('Tr_velo_to_cam: ', Tr_velo_to_cam)
    # calib_file = os.path.join(calib_path, '7-112.txt')
    print(os.path.dirname(calib_file))
    if not os.path.exists(os.path.dirname(calib_file)):
        os.makedirs(os.path.dirname(calib_file))

    with open(calib_file, 'w') as f:
        f.write('P0: %s\n' % (str(P0).replace(',', ' ')[1:-1]))
        f.write('Tr_velo_to_cam: %s\n' % (str(Tr_velo_to_cam).replace(',', ' ')[1:-1]))
    print('*******save to %s*********' % calib_file)


if __name__ == '__main__':
    mydict = {
          'camera_0_0': 's0_0',
          'camera_1_0': 's1_0',
          'camera_2_0': 's2_0',
          'camera_3_0': 's3_0',
          'camera_0_8': 's0_8',
          'camera_1_8': 's1_8',
          'camera_2_8': 's2_8',
          'camera_3_8': 's3_8',
          }

    # intrin_yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-25北京环贸纯视觉bev路口/6号路口/input/cam'
    # yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-25北京环贸纯视觉bev路口/6号路口/output'
    # intrin_yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与兴华路/input/cam'
    # yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与兴华路/output'
    # intrin_yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-25北京环贸纯视觉bev路口/7号路口/input/cam'
    # yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-25北京环贸纯视觉bev路口/7号路口/output'
    intrin_yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与创业四路/input/cam'
    yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与创业四路/output'
    
    res = []
    for i, path in enumerate(mydict.keys()):
        calib_file_basename = f'cam_ground_extrinsic_calib_result_{mydict[path]}'
        intrin_yaml_file = os.path.join(intrin_yaml_path, calib_file_basename + '.yaml')
        yaml_file = os.path.join(yaml_path, calib_file_basename + '_common.yaml')
        res+=get_bev_param(yaml_file, cam_num=i, old_calib_version=True, intrinsic_yaml_file=intrin_yaml_file)
