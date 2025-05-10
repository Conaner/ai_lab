import os
import pandas as pd

def image_emotion_mapping(path):
    # 读取 emotion 文件
    df_emotion = pd.read_csv('/root/autodl-tmp/faceExpression/data/process/emotion.csv', header=None)
    
    # 查看该文件夹下的所有文件
    files_dir = os.listdir(path)
    
    # 用于存储图像路径
    path_list = []
    # 用于存储图像对应的情感标签
    emotion_list = []
    
    # 遍历该文件夹下的所有文件
    for file_dir in files_dir:
        # 如果某个文件是图像文件，则将其文件名和对应的情感标签分别添加到 path_list 和 emotion_list 中
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            emotion_list.append(df_emotion.iat[index, 0])
    
    # 将两个列表合并成一个 DataFrame 并保存为 image_emotion.csv 文件
    path_s = pd.Series(path_list)
    emotion_s = pd.Series(emotion_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['emotion'] = emotion_s
    df.to_csv(os.path.join(path, 'image_emotion.csv'), index=False, header=False)

def main():
    # 指定文件夹路径
    train_set_path = '/root/autodl-tmp/faceExpression/data/pre_process/train_set'
    verify_set_path = '/root/autodl-tmp/faceExpression/data/pre_process/verify_set'
    
    # 创建训练集和验证集的映射文件
    image_emotion_mapping(train_set_path)
    image_emotion_mapping(verify_set_path)

if __name__ == "__main__":
    main()