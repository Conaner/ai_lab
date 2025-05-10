import os
import random
import shutil

def split_data(src_image_dir, src_label_dir, dst_train_dir, dst_val_dir, dst_test_dir, split_ratio=(0.7, 0.2, 0.1)):
    # 创建目标目录
    os.makedirs(dst_train_dir, exist_ok=True)
    os.makedirs(dst_val_dir, exist_ok=True)
    os.makedirs(dst_test_dir, exist_ok=True)

    # 获取所有图像文件名
    image_files = [f for f in os.listdir(src_image_dir) if f.endswith('.png')]

    # 计算划分的数量
    total_count = len(image_files)
    train_count = int(total_count * split_ratio[0])
    val_count = int(total_count * split_ratio[1])
    test_count = total_count - train_count - val_count

    # 随机打乱图像文件列表
    random.shuffle(image_files)

    # 划分数据集
    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]
    test_images = image_files[train_count + val_count:]

    # 定义复制函数
    def copy_files(image_list, dst_image_dir, dst_label_dir):
        for image_file in image_list:
            label_file = os.path.splitext(image_file)[0] + '.txt'
            src_image_path = os.path.join(src_image_dir, image_file)
            src_label_path = os.path.join(src_label_dir, label_file)
            dst_image_path = os.path.join(dst_image_dir, image_file)
            dst_label_path = os.path.join(dst_label_dir, label_file)
            shutil.copy2(src_image_path, dst_image_path)
            shutil.copy2(src_label_path, dst_label_path)

    # 复制训练集
    copy_files(train_images, dst_train_dir, dst_train_dir)
    print(f"Copied {len(train_images)} images and labels to {dst_train_dir}")

    # 复制验证集
    copy_files(val_images, dst_val_dir, dst_val_dir)
    print(f"Copied {len(val_images)} images and labels to {dst_val_dir}")

    # 复制测试集
    copy_files(test_images, dst_test_dir, dst_test_dir)
    print(f"Copied {len(test_images)} images and labels to {dst_test_dir}")

# 指定源目录和目标目录
src_image_dir = '\pythonpa\ch01-1\FaceMask\maskDet\data\orignalface\images'
src_label_dir = '\pythonpa\ch01-1\FaceMask\maskDet\data\orignalface\labels'
dst_train_dir = '\pythonpa\ch01-1\FaceMask\maskDet\data\train'
dst_val_dir = '\pythonpa\ch01-1\FaceMask\maskDet\data\val'
dst_test_dir = '\pythonpa\ch01-1\FaceMask\maskDet\data\test'
split_ratio = (0.7, 0.2, 0.1)

# 调用函数进行数据集划分
split_data(src_image_dir, src_label_dir, dst_train_dir, dst_val_dir, dst_test_dir, split_ratio)