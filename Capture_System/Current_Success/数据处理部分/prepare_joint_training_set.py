import os
import shutil
from pathlib import Path


# 仅用于“联合训练前的数据准备”，不修改原训练/预测脚本
SOURCE_DIRS = [
    r"D:\nightmare\Documents\SRTP\AllData\Processed_Final_Training_Set_1",
    r"D:\nightmare\Documents\SRTP\AllData\Processed_Final_Training_Set_1.14_1",
]
OUTPUT_DIR = r"D:\nightmare\Documents\SRTP\AllData\Processed_Final_Training_Set_1_plus_1.14_1"


def is_trainable_file(path: Path) -> bool:
    # 训练脚本会读取 .csv/.xlsx，但当前目录里有 *_predicted.csv（非训练标签格式）
    # 为避免混入预测结果，这里只汇总原始 .xlsx 文件。
    return path.suffix.lower() == ".xlsx"


def source_tag(src_dir: str) -> str:
    # 用目录名做前缀，避免同名文件覆盖（例如 Sub_10... 在两个目录都存在）
    return Path(src_dir).name.replace(" ", "_")


def main():
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for src in SOURCE_DIRS:
        src_path = Path(src)
        if not src_path.exists():
            print(f"[跳过] 源目录不存在: {src_path}")
            continue

        tag = source_tag(src)
        for file_path in src_path.iterdir():
            if not file_path.is_file():
                continue
            if not is_trainable_file(file_path):
                skipped += 1
                continue

            new_name = f"{tag}__{file_path.name}"
            target_path = output / new_name
            shutil.copy2(file_path, target_path)
            copied += 1

    print("=" * 60)
    print(f"合并完成，输出目录: {output}")
    print(f"复制的训练文件数(.xlsx): {copied}")
    print(f"跳过的非训练文件数: {skipped}")
    print("=" * 60)
    print("下一步：将训练脚本 CONFIG['data_dir'] 指向该输出目录进行训练。")


if __name__ == "__main__":
    main()
