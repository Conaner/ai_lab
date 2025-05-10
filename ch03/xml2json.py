import os
import json
import xmltodict
 
def xml_to_JSON(xml):
    # 格式转换
    try:
        convertJson = xmltodict.parse(xml, encoding = 'utf-8')
        jsonStr = json.dumps(convertJson, indent=1)
        return jsonStr
    except Exception:
        print('something has occurred')
        pass
 
def find_read_list(path):
    # 获取该文件夹下所有以.xml为后缀的文件
    file_list = os.listdir(path)
    read_list = []
    for i in file_list:
        a, b = os.path.splitext(i)
        if b == '.xml':
            read_list.append(i)
        else:
            continue
    return read_list
 
def batch_convert(path):
    # 主函数
    in_list = find_read_list(path)
    print(in_list)
    for item in in_list:
        with open(path+'\\'+item, encoding = 'utf-8') as f:
            xml = f.read()
            converted_doc = xml_to_JSON(xml)
        new_name = item.rsplit('.xml')[0] + '.json'
        with open(path+'\\'+new_name, 'w+',encoding = 'utf-8') as f:
            f.write(converted_doc)
            print('{} has finished'.format(new_name))
 
# 在这边输入文件夹路径，接下来就会把这个文件夹下所有以.xml为后缀的文件转换为.json文件
# 注意Python文件路径的输入格式 \\
batch_convert(r'C:\Users\17216\Desktop\ai_lab\ch03\train\xml')