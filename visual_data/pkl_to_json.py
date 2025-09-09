import os
import json
import pickle
import numpy as np
from argparse import ArgumentParser

def convert_pkl_to_json(pkl_path, json_path=None, indent=4, ensure_ascii=False):
    """
    将 pkl 文件转换为可视化的 JSON 文件
    
    参数:
        pkl_path (str): 输入 pkl 文件路径
        json_path (str): 输出 json 文件路径，默认与 pkl 同目录同名称
        indent (int): JSON 缩进空格数，增强可读性
        ensure_ascii (bool): 是否确保 ASCII 编码，False 可保留中文等字符
    """
    # 读取 pkl 文件
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"读取 pkl 文件失败: {e}")

    # 处理 numpy 数据类型（转换为 Python 原生类型）
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            # 数组转换为列表
            return obj.tolist()
        elif isinstance(obj, np.generic):
            # 单个 numpy 元素（如 np.int64）转换为 Python 类型
            return obj.item()
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            # 时间类型转换为字符串
            return str(obj)
        elif hasattr(obj, '__dict__'):
            # 类实例转换为字典（仅保留属性）
            return obj.__dict__
        else:
            # 其他类型尝试直接序列化（JSON 不支持的类型会报错）
            return obj

    # 生成输出路径
    if not json_path:
        json_path = os.path.splitext(pkl_path)[0] + '.json'

    # 写入 JSON 文件
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                default=serialize  # 处理特殊类型
            )
        print(f"转换成功！JSON 文件已保存至: {json_path}")
    except Exception as e:
        raise ValueError(f"写入 JSON 文件失败: {e}")

if __name__ == '__main__':
    # 命令行参数解析
    parser = ArgumentParser(description='将 pkl 文件转换为可视化 JSON 文件')
    parser.add_argument('pkl_path', help='输入 pkl 文件的路径')
    parser.add_argument('-o', '--output', help='输出 JSON 文件的路径（可选）')
    parser.add_argument('-i', '--indent', type=int, default=4, help='JSON 缩进空格数（默认 4）')
    args = parser.parse_args()

    # 调用转换函数
    convert_pkl_to_json(
        pkl_path=args.pkl_path,
        json_path=args.output,
        indent=args.indent
    )