from vis_anno_pipeline import *

def vis_anno_pipeline(config):
    data_root = config.data_root
    dst_root_dir = config.dst_root_dir
    pcd_source_file = config.pcd_source_file
    intrin_yaml_path = config.intrin_yaml_path
    yaml_path = config.yaml_path
    road_info = config.road_info
    img_dirs = config.img_dirs
    calib_dict = config.calib_dict

    # ================== 缺省相机检测 ========================
    camera_check = [1] * 8
    for i in range(4):
        if f'camera_{i}_0' not in img_dirs:
            camera_check[i] = 0
        if f'camera_{i}_8' not in img_dirs:
            camera_check[i+4] = 0

    # DAN 可上题文件夹
    dan_dst_dir = os.path.join(dst_root_dir,'dan')
    dan_dst_img_dir = os.path.join(dan_dst_dir,'send/1')
    dst_kitti_root = os.path.join(dst_root_dir, 'kitti_result')
    txt_name = os.path.basename(dst_root_dir)
    txt_path = os.path.join(dst_kitti_root, txt_name +'.txt')

    # =================== 1. 拷贝 adp格式 图像数据 无间隔 ====================
    print("1. 拷贝 adp格式 图像数据 无间隔")
    print(f"Copying file from {data_root} to {dan_dst_img_dir}")
    data_dirs = sorted(os.listdir(data_root))
    
    all_img_list = []
    for data_dir in tqdm(data_dirs):
        data_dir_path = os.path.join(data_root, data_dir)
        if not os.path.isdir(os.path.join(data_root, data_dir)):
            continue
        if not set(img_dirs).issubset(set(os.listdir(data_dir_path))):
            continue
        # 从第10帧开始，保证每帧数据都之后都能用作时序数据
        # if len(os.listdir(os.path.join(data_dir_path, img_dirs[0])))<=10:
        #     continue
        img_list_tmp = os.listdir(os.path.join(data_dir_path, img_dirs[0]))
        all_img_list += img_list_tmp

    # 按照 5s 间隔，过滤数据
    # filter_img_list = filter_timestamps(all_img_list)

    # 准备任务列表
    tasks = []
    for data_dir in tqdm(data_dirs, desc='preparing tasks'):
        if not os.path.isdir(os.path.join(data_root, data_dir)):
            continue
        if not set(img_dirs).issubset(set(os.listdir(os.path.join(data_root, data_dir)))):
            continue
        for img_dir in img_dirs:
            tasks.append((data_dir, img_dir))

    # 开多线程执行任务
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(copy_image, data_root, data_dir, img_dir, all_img_list, dan_dst_img_dir, road_info) for data_dir, img_dir in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
    
    # # =========================== 2. 将目标文件写入txt ===================================
    print("2. 将目标文件写入txt")
    os.makedirs(dst_kitti_root, exist_ok=True)
    
    # generate_txt_from_filenames(os.path.join(dan_dst_img_dir, img_dirs[0]), txt_path)
    generate_txt_from_common_filenames(dan_dst_img_dir, txt_path, dirs=img_dirs)

    # ========================== 3. 生成kitti格式数据 ============================
    print("3. 生成kitti格式数据")
    src_data_root = dan_dst_img_dir
    src_img_dir = src_data_root
    dst_img_dir = os.path.join(dst_kitti_root, "images")

    # 读取文件名列表
    with open(txt_path, 'r') as f:
        target_filename = [line.strip() for line in f]

    # 确保目标目录存在
    os.makedirs(dst_img_dir, exist_ok=True)

    for i in range(4):
        os.makedirs(os.path.join(dst_img_dir, f'camera_{i}'), exist_ok=True)
        os.makedirs(os.path.join(dst_img_dir, f'camera_{i}_fisheye'), exist_ok=True)

    # 单个文件的复制逻辑
    def copy_file(file):
        for i in range(4):
            # 普通相机图像
            img_name = file + '.jpg'
            img_name_dst = file +'.jpg'

            src_gun_img_path = os.path.join(src_img_dir, img_dirs[i], img_name)
            dst_gun_img_path = os.path.join(dst_img_dir, f'camera_{i}', img_name_dst)
            # print(f"copying {src_gun_img_path} to {dst_gun_img_path}")
            if camera_check[i]==1:
                shutil.copy(src_gun_img_path, dst_gun_img_path)
            else:
                # 创建一张纯黑图像代替
                black_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.imwrite(dst_gun_img_path, black_img)

            # 鱼眼相机图像
            src_fish_img_path = os.path.join(src_img_dir, f'camera_{i}_8', img_name)
            dst_fish_img_path = os.path.join(dst_img_dir, f'camera_{i}_fisheye', img_name_dst)
            if camera_check[i+4]==0:
                # 创建一张纯黑图像代替
                black_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.imwrite(dst_fish_img_path, black_img)
                return

            try:
                fish_img = cv2.imread(src_fish_img_path)
                if fish_img is None:
                    raise ValueError(f"cv2.imread failed to read image: {src_fish_img_path}")
                fish_img = cv2.resize(fish_img, (720, 720))
            except Exception as e:
                print(f"[Error] Failed to process image: {src_fish_img_path}", file=sys.stderr)
                print(f"Exception: {repr(e)}", file=sys.stderr)
                raise  # 继续抛出，方便调试或上层捕捉
            left = (1280 - 720) // 2
            right = 1280 - 720 - left
            top = bottom = 0
            fish_img = cv2.copyMakeBorder(fish_img, top, bottom, left, right,
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
            cv2.imwrite(dst_fish_img_path, fish_img)


    # 使用线程池并行复制
    with ThreadPoolExecutor(max_workers=30) as executor:
        list(tqdm(executor.map(copy_file, target_filename), total=len(target_filename)))
    
    print('✅ All images copied.')

    # =================== 4. 生成标定文件calib_bev.txt =====================
    calib_save_path = os.path.join(dst_kitti_root, 'calib_bev.txt')
    
    res = []
    for i, path in enumerate(calib_dict.keys()):
        if calib_dict[path] is not None:
            calib_file_basename = f'cam_ground_extrinsic_calib_result_{calib_dict[path]}'
            intrin_yaml_file = os.path.join(intrin_yaml_path, calib_file_basename + '.yaml')
            yaml_file = os.path.join(yaml_path, calib_file_basename + '_common.yaml')
            res+=get_bev_param(yaml_file, cam_num=i, old_calib_version=True, intrinsic_yaml_file=intrin_yaml_file)
        else:
            # 该相机不存在，写入单位矩阵
            res+=['P{}: 1 0 0 0 0 1 0 0 0 0 1 0'.format(i)]
            res+=['Tr_velo_to_cam{}: 1 0 0 0 0 1 0 0 0 0 1 0'.format(i)]

    with open(calib_save_path, 'w') as f:
        for line in res:
            f.write(line + '\n')

    calib_source_file = os.path.join(dst_kitti_root, 'calib_bev.txt')
    target_calib_dir = os.path.join(dst_kitti_root, 'calib')

    file_names = []
    print('preparing file names...')
    with open(txt_path, 'r') as f:
        # i10_sift_filename = [line.strip() for line in f]
        file_names = [line.strip() for line in f]

    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_calib_dir):
        os.makedirs(target_calib_dir)

    # 读取calib.txt文件内容
    with open(calib_source_file, 'r') as f:
        calib_content = f.read()

    print('generating calib file...')
    for file_name in tqdm(file_names):
        base_name = file_name
        target_calib_file = os.path.join(target_calib_dir, f'{base_name}.txt')

        with open(target_calib_file, 'w') as f:
            f.write(calib_content)

    print('Calibration files generation completed.')
    
    # ==================== 5. 创建空label ==============================
    target_label_dir = os.path.join(dst_kitti_root, 'label')
    os.makedirs(target_label_dir, exist_ok=True)
    files_list = file_names

    for file in tqdm(files_list, desc='创建空label'):
        file = file + '.txt'
        dst_path = os.path.join(target_label_dir, file)
        with open(dst_path, 'w') as f:
            pass
    # =================== 6. 生成数据集 pkl ==================================
    print('创建 pkl')
    # cmd = f"python /adga/lushiyong/bevformer_road_8cam/tools/create_pkl/create_data.py --root-path {dst_kitti_root} --txt-name {txt_name}.txt --output-pkl-name {txt_name}.pkl"
    cmd = f"python bevformer-roadside/tools/create_pkl/create_data.py --root-path {dst_kitti_root} --txt-name {txt_name}.txt --output-pkl-name {txt_name}.pkl"

    # 捕获输出
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")

    for line in process.stdout:
        print(line, end="") 
    
    # =================== 7. 通过pkl进行推理, 关键步骤！ ==================================
    print('开始推理！')
    # CONFIG_PATH='/adga/lushiyong/bevformer_road_8cam/projects/configs/bevformer_road_mogo_lsy/test_save_query_bevformer_r101_hyi7_i10_mix_pretrain_200m_200_scale_0.7.py'
    # CHECKPOINT_PATH='/adga/lushiyong/bevformer_road_8cam/work_dirs/mogobev/bevformer_base_200m_300_mix_all_77187_sampler/epoch_50.pth'
    # TXT_SAVE_PATH=f'/adga/lushiyong/bev_8cam_compare_result/4d_pipeline/{txt_name}/conf0.3_ep50'
    CONFIG_PATH='projects/configs/bevformer_road_mogo_lsy/test_save_query_bevformer_r101_hyi7_i10_mix_pretrain_200m_200_scale_0.7.py'
    CHECKPOINT_PATH='work_dirs/mogobev/bevformer_base_200m_300_mix_all_77187_sampler/epoch_50.pth'
    TXT_SAVE_PATH=f'/data1/turbo_data/huben/tmp/vision_only_4d_pipeline/{txt_name}/conf0.3_ep50'
    DATA_ROOT=dst_kitti_root
    PKL_PATH=os.path.join(DATA_ROOT, f'{txt_name}.pkl')
    DST_DIR=dst_kitti_root
    query_save_path = os.path.join(DST_DIR, 'query_feats')

    # cmd = f"cd /adga/lushiyong/bevformer_road_8cam && python tools/test/4d_inference_pipeline.py --config {CONFIG_PATH} --checkpoint {CHECKPOINT_PATH} --save-path {TXT_SAVE_PATH} --data-root {DATA_ROOT} --ann-file {PKL_PATH} --query-save-path {query_save_path} && cd - && cp -r {TXT_SAVE_PATH}/3d_url {DST_DIR}/infer_result"
    cmd = f"cd bevformer-roadside && export PYTHONPAHT=./ && python tools/test/4d_inference_pipeline.py --config {CONFIG_PATH} --checkpoint {CHECKPOINT_PATH} --save-path {TXT_SAVE_PATH} --data-root {DATA_ROOT} --ann-file {PKL_PATH} --query-save-path {query_save_path} && cd - && cp -r {TXT_SAVE_PATH}/3d_url {DST_DIR}/infer_result && cd .."
    # cmd = f"python bevformer-roadside/tools/test/4d_inference_pipeline.py --config {CONFIG_PATH} --checkpoint {CHECKPOINT_PATH} --save-path {TXT_SAVE_PATH} --data-root {DATA_ROOT} --ann-file {PKL_PATH} --query-save-path {query_save_path} && cd - && cp -r {TXT_SAVE_PATH}/3d_url {DST_DIR}/infer_result"
    print(cmd)
    # # 捕获输出
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    for line in process.stdout:
        print(line, end="") 

    # ===================== 插一个 过滤query_feats ============================
    print('过滤query_feats')
    import pickle
    query_feats_dir = os.path.join(dst_kitti_root, 'query_feats')
    filtered_query_feats_dir = os.path.join(dst_kitti_root, 'query_feats')
    os.makedirs(filtered_query_feats_dir, exist_ok=True)
    for pkl_file in os.listdir(query_feats_dir):
        if pkl_file.endswith('.pkl'):
            pkl_path = os.path.join(query_feats_dir, pkl_file)
            infer_res_path = os.path.join(dst_kitti_root, 'infer_result', pkl_file.replace('.pkl', '.txt'))
            with open(infer_res_path, 'r') as f:
                lines = f.readlines()
            
            line_data = np.array([line.strip().split(' ') for line in lines])
            line_data = np.delete(line_data, np.s_[1:8], axis=1)
            
            line_data[:, -2] = np.float32(line_data[:, -2])/3.14*180.0
            target_num = len(lines)
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            for i in range(len(data)):
                data[i] = data[i][:target_num]

            query_feats = data[0]
            final_res = np.hstack((line_data, query_feats))

            filtered_pkl_path = os.path.join(filtered_query_feats_dir, pkl_file)
            with open(filtered_pkl_path, 'wb') as f:
                pickle.dump(final_res, f)
    # ===================== 8. nms + 类别置信度过滤 ================================
    label_dir = f"{dst_kitti_root}/infer_result"      # 输入文件夹
    output_dir = f"{dst_kitti_root}/infer_result_nms" # 输出文件夹
    process_labels(label_dir, output_dir, iou_thr=0.1, num_workers=20)
    
    # ===================== 9. txt 2 json 转到 DAN类别 ============================
    from txt2json_convert import process_files_in_parallel
    save_dir = f'{dan_dst_dir}/result_json/'
    files_list = os.listdir(output_dir)

    process_files_in_parallel(output_dir, save_dir, files_list)

    # ===================== 10. 复制伪点云 ==========================================
    source_image_dir = save_dir
    target_pcd_dir = f'{dan_dst_img_dir}/3d_url/'

    if not os.path.exists(target_pcd_dir):
        os.makedirs(target_pcd_dir)

    with open(pcd_source_file, 'r') as f:
        content = f.read()

    # 获取源目录中的所有文件名（假设都是图像文件）
    file_names = []
    for f in tqdm(os.listdir(source_image_dir)):
        if os.path.isfile(os.path.join(source_image_dir, f)):
            file_names.append(f)

    for file_name in tqdm(file_names, desc='Created pseudo pcd'):
        base_name, _ = os.path.splitext(file_name)
        target_file = os.path.join(target_pcd_dir, f'{base_name}.pcd')

        with open(target_file, 'w') as f:
            f.write(content)

    print('伪点云复制完毕')

    return dst_kitti_root

def vis_4d_pipeline(config):
    dst_kitti_root = vis_anno_pipeline(config)

    # ================================= 跟踪 =================================
    # detection_result_path = os.path.join(dst_kitti_root, 'infer_result')
    # dst_track_result_path = os.path.join(dst_kitti_root, 'track_result')

    # cmd = f"./track.sh {detection_result_path} {dst_track_result_path}"
    # print(cmd)
    # # # 捕获输出
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    # for line in process.stdout:
    #     print(line, end="")

    # =============================== refine =================================
    # refine_result_path = os.path.join(dst_kitti_root, 'refine_result')
    


if __name__ == "__main__":
    # TODO: select config by road_id!!!
    import config.hm_7 as config

    dst_kitti_root = vis_4d_pipeline(config)