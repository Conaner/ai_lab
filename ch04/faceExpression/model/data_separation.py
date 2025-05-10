# emotion和pixels 数据
import pandas as pd

path='/root/autodl-tmp/faceExpression/data/raw/train.csv'

# 读取数据
df= pd.read_csv(path)

# 提取 emotion 数据
df_y= df[['emotion']]

# 提取 pixels 数据
df_x= df[['pixels']]

# 将 emotion 数据保存为 emotion.csv
df_y.to_csv('/root/autodl-tmp/faceExpression/data/process/emotion.csv', index=False, header=False)

# 将 pixels 数据保存为 pixels.csv
df_x.to_csv('/root/autodl-tmp/faceExpression/data/process/pixels.csv', index=False, header=False)