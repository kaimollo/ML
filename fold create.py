import os, shutil


original_dataset_dir = 'C:/Users/hasee/Desktop/dogs-vs-cats/train/train'    # 原始文解压目录
base_dir = 'D:\JAVA0\pacharm\cat-dog classification/dataset'     #保存较小数据集目录
os.mkdir(base_dir)  #创建新的文件夹


# 分别对应划分好的训练，验证和测试目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#猫的训练目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# 狗的训练目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# 猫的验证目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# 狗的验证目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# 猫的测试目录
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# 狗的测试目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
# 将前6000张猫的图像复制到train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(6000)]  # format函数通过{}来指点字符串处理的位置，储存为列表形式
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)  # copyfile实现将一个文件中的内容复制道另一个文件中去，src是来源文件；dst是目标文件

# 将剩下的2000张图像复制到validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(6000, 8000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来2000张图片复制到test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(8000, 10000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将前6000张狗的图片复制到train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(6000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来2000张图像复制到validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(6000, 8000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来2000张图像复制到test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(8000, 10000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))    #os.listdir列举指定目录中的文件名
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))