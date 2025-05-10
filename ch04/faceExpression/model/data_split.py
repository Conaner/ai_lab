import os
import shutil

def create_directories():
    # 创建目标文件夹
    train_folder = '/root/autodl-tmp/faceExpression/data/pre_process/train_set'
    verify_folder = '/root/autodl-tmp/faceExpression/data/pre_process/verify_set'
    
    # 如果文件夹不存在，则创建
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(verify_folder):
        os.makedirs(verify_folder)
    
    return train_folder, verify_folder

def split_data(src_folder, train_folder, verify_folder):
    # 获取所有图片文件名
    files = os.listdir(src_folder)
    
    # 将图片按顺序划分到训练集和验证集中
    for i, file_name in enumerate(files):
        src_path = os.path.join(src_folder, file_name)
        if i < 24000:
            # 将前24000张图片放入训练集
            dst_path = os.path.join(train_folder, file_name)
        else:
            # 将剩余图片放入验证集
            dst_path = os.path.join(verify_folder, file_name)
        
        # 复制文件
        shutil.copy(src_path, dst_path)

def main():
    # 源文件夹路径
    src_folder = '/root/autodl-tmp/faceExpression/data/process/images'
    
    # 创建目标文件夹
    train_folder, verify_folder = create_directories()
    
    # 划分数据集
    split_data(src_folder, train_folder, verify_folder)

if __name__ == "__main__":
    main()