import pickle
pkl_path = "/adga/lushiyong/vis_anno/dataset/4d/train_sh_3d_road_2_20250531_5000_lx/kitti_result/query_feats/1748647643100_sh2.pkl"
# pkl_path = "/adga/lushiyong/vis_anno/dataset/4d/train_sh_3d_road_2_20250531_5000_lx/kitti_result/filtered_query_feats/1748647643100_sh2.pkl"

with open(pkl_path, "rb") as f:
    loaded_list = pickle.load(f)

print(loaded_list)