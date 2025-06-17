import argparse

from BEVFUSION.tools.data_converter import mogo_converter_wrh as mogo_converter

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/data1/turbo_data/RALG/data/3.0_pro/Intersection/version/v4_fisheye/',type=str)
    parser.add_argument('--use_fisheye', action='store_true', help="whether use fisheye")
    args = parser.parse_args()

    mogo_data_prep(
        root_path=args.data_root, 
        info_prefix="mogo", 
        version="1.0",
        load_augmented=None,
        use_fisheye=args.use_fisheye,  # 路口需要鱼眼
    )