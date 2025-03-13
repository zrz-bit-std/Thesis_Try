import sys
sys.path.append('/data1/turbo_data/wangruihao/code/auto_labeling/4D_label')
from parse_data.parse_data_for_bevpro import mogo_converter_wrh as mogo_converter


def mogo_data_prep(
    root_path,
    info_prefix,
    version,
    load_augmented=None,
    use_fisheye = False
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.
        # generate pkl !!!
        mogo_converter.create_mogo_infos(
            root_path, info_prefix, version=version, use_fisheye = use_fisheye
        )

if __name__ == "__main__":
    import os
    save_dir = '/data1/turbo_data/4D_label_dataset/models_res/bevpro'
    sub_t = 'train_hy_4d_road_7_20250211_lx'
    data_root = os.path.join(save_dir,sub_t)
    mogo_data_prep(root_path = data_root, 
                    info_prefix = "mogo", 
                    version ="1.0",
                    load_augmented=None,
                    use_fisheye = True)