'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-02 08:59:12
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-03 09:41:36
FilePath: \newrgzn\rename_files.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import random
import shutil

# 设置原始图片文件夹和目标文件夹
source_folder = 'download'
target_folder = 'trans'

# 创建目标文件夹和子文件夹
os.makedirs(os.path.join(target_folder, 'trains', 'bread'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'trains', 'dog'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'validation', 'bread'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'validation', 'dog'), exist_ok=True)

# 获取原始文件夹中的子文件夹（狗和面包）
subfolders = os.listdir(source_folder)

for subfolder in subfolders:
    source_subfolder = os.path.join(source_folder, subfolder)
    target_train_bread = os.path.join(target_folder, 'trains', 'bread')
    target_train_dog = os.path.join(target_folder, 'trains', 'dog')
    target_validation_bread = os.path.join(target_folder, 'validation', 'bread')
    target_validation_dog = os.path.join(target_folder, 'validation', 'dog')

    # 获取子文件夹中的所有图片文件
    image_files = os.listdir(source_subfolder)
    random.shuffle(image_files)

    # 计算前一半图片的数量
   
    print(len(image_files))
    print(len(image_files)//2)
    half_count = len(image_files) // 2

    # 将前一半图片移动到训练集文件夹
    for image_file in image_files[:half_count]:
        source_image_path = os.path.join(source_subfolder, image_file)
        if subfolder == '狗':
            target_image_path = os.path.join(target_train_dog, image_file)
        else:
            target_image_path = os.path.join(target_train_bread, image_file)
        shutil.move(source_image_path, target_image_path)

    # 将后一半图片移动到验证集文件夹
    for image_file in image_files[half_count:]:
        source_image_path = os.path.join(source_subfolder, image_file)
        if subfolder == '狗':
            target_image_path = os.path.join(target_validation_dog, image_file)
        else:
            target_image_path = os.path.join(target_validation_bread, image_file)
        shutil.move(source_image_path, target_image_path)

print("分割完成")

folder_paths = [
    'trans/trains/dog',
    'trans/trains/bread',
    'trans/validation/dog',
    'trans/validation/bread'
]

for folder_path in folder_paths:
    # 获取该文件夹下的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # 初始化序号
    counter = 1
    
    # 遍历图片文件并重命名
    for old_name in image_files:
        # 构建新的文件名
       
        new_name = f'{os.path.basename(folder_path)}.{counter}.jpg'
        # 构建旧文件的完整路径和新文件的完整路径
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(old_path, new_path)
        
        # 增加序号
        counter += 1

print('重命名完成')