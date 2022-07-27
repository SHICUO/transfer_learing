# coding:utf-8
# @Time:2022/7/14 16:03
# @Author:LHT
# @File:transfer_learning
# @GitHub:https://github.com/SHICUO
# @Contact:lin1042528352@163.com
# @Software:PyCharm

# tf.Session https://blog.csdn.net/qq_36201400/article/details/108345169

import os
import random
import re
import sys
import tarfile
import urllib.request
from datetime import datetime
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import tensorflow as tf
import hashlib
import logging
import argparse


FLAGS = None
"""
These are all parameters that are tied to the particular model architecture
we're using for Inception v3. These include things like tensor names and their
sizes. If you want to adapt this script to work with another model, you will
need to update these to reflect the values in the network you're using.
"""
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # 134M


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """构造训练数据集，包括train、test、valid

    gfile https://zhuanlan.zhihu.com/p/31536538
    logging https://zhuanlan.zhihu.com/p/425678081
    hashlib https://docs.python.org/zh-cn/3/library/hashlib.html

    :param image_dir: 图片目录路径。格式：flower_photos/rose/anotherphoto77.jpg，输入flower_photos
    :param testing_percentage: 测试集大小百分比 如输入50,代表计算得到哈希值占图片哈希值域(MAX_NUM_IMAGES_PER_CLASS)小于50的图片集合
    :param validation_percentage: 验证集大小百分比

    :return: 返回的数据集占比会近似为 1-testing_percentage-validation_percentage：testing_percentage：validation_percentage
    """
    if not gfile.Exists(image_dir):
        logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]  # 递归获取路径里目录名
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:     # 去掉图片路径里的根目录,根目录是第一个
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'png', 'bmp']
        file_list = []
        dir_name = os.path.basename(sub_dir)  # 分类目录名称
        logging.info("Look for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, "*."+extension)
            # 查找符合 file_glob 形式的文件名，并添加到 file_list 中
            file_list.extend(gfile.Glob(file_glob))  # len=2000
        if not file_list:
            logging.warning('No files found')
            continue
        if len(file_list) < 20:     # 图片太少
            logging.warning('WARNING: Folder has less then 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            logging.warning("WARNING: Folder {} has more than {} images. Some images will never be selected.".format(
                dir_name, MAX_NUM_IMAGES_PER_CLASS))
        # lower() 返回B的副本，并将所有ASCII字符转换为小写。
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())  # 除了a-z和0-9其他字符全部替换为空格字符
        if label_name == ' ':
            label_name = dir_name
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            # 匹配到 _nohash_后面加一个或多个除换行符 \n 之外的任何单字符
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            # print(file_name, hash_name)
            # as_bytes 将二进制或unicode编码的字符串转换为字节编码字符串，对文本使用utf-8编码。
            # hexdigest() 返回哈希摘要，作为十六进制数据字符串值
            # print(hash_name)
            # print(compat.as_bytes(hash_name))
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            # print(hash_name_hashed)
            # 将 hash_name_hashed 装换为16进制整型
            # 哈希百分比，当前name哈希编码占总编码范围的大小
            percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS+1))
                               * (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            # print(percentage_hash)
            # 根据哈希值占比去分配训练、测试、验证集，整个过程体现了随机性
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # 添加 label_name 当前类别数据集信息
        result[label_name] = {
            'dir': dir_name,    # 文件夹名字，如cats，dogs
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }

    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """根据给定的index返回对应的图像路径

    :param image_lists: 每个label图片的字典
    :param label_name: 图片的label
    :param index: 图像的整数偏移量，通过label的图像数量进行取模，可以是任意大的
    :param image_dir: 包含图片的子文件的根文件夹路径
    :param category: 从数据集中提取的数据用途类别，是train、test、还是valid

    :return: 满足条件的一张图片的文件路径
    """
    if label_name not in image_lists:   # 判断图片类别是否存在
        logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:     # 判断图片是否划分了数据集
        logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:              # 判断图片是否为空
        logging.fatal('Label %s has no images in the category %s.', label_name, category)
    mod_index = index % len(category_list)  # 索引除以图片数量
    base_name = category_list[mod_index]    # 图片文件名
    sub_dir = label_lists['dir']            # 包含图片的子文件夹名
    full_path = os.path.join(image_dir, sub_dir, base_name)    # 图片的全路径
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture):
    """根据给定索引的的图片数据缓存的txt文件的文件路径

    :param image_lists:
    :param label_name:
    :param index:
    :param bottleneck_dir: 保存 bottleneck 值缓存文件的文件夹
    :param category:
    :param architecture: 使用的模型

    :return:
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + architecture + '.txt'


def create_model_graph(model_info):
    """从保存的GraphDef文件创建一个graph，并返回一个graph对象。

    :param model_info: 包含模型体系结构信息的字典。

    :return: 返回 Inception 网络的 Graph，还有各种需要处理的张量
    """
    # 生成新计算图 graph ，并设为默认图
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        # gfile.FastGFile() 类似于python提供的文本操作open()函数，无阻塞速度快
        with gfile.FastGFile(model_path, 'rb') as f:
            # graph_def = tf.compat.v1.GraphDef()
            graph_def = graph.as_graph_def()
            # 对模型f读取到的字符串数据进行解析并存放到graph_def
            graph_def.ParseFromString(f.read())
            # tf.import_graph_def() 获得 return_elements 指定部分的“Operation”和“Tensor”对象列表
            bottleneck_tensor, resized_input_tensor = tf.import_graph_def(
                graph_def, name='', return_elements=[   # name是前缀
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]
            )

    return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor,
                            resized_input_tensor, bottleneck_tensor):
    """在图片运行推断，提取 bottleneck layer 上的结果

    :param sess: 当前活动的TensorFlow会话
    :param image_data: 图片数据
    :param image_data_tensor: 输入数据的操作
    :param decoded_image_tensor: 初始图像进行大小调整和预处理的操作
    :param resized_input_tensor: 数据输入图节点的操作
    :param bottleneck_tensor: 在最后的softmax之前的层的操作

    :return: Numpy array of bottleneck values.
    """
    # 进行推理计算 y=x*2 run(y, {x: x_in}) x,y是占位符， x_in是真正的输入数据
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)   # linear层的输出 2048维的向量降维
    return bottleneck_values


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):  # exists不区分大小写
        os.makedirs(dir_name)


def maybe_download_and_extract(data_url):
    """下载并提取模型tar文件。

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    --tarfile https://www.liujiangblog.com/course/python/63

    :param data_url: 预训练模型的web网址
    """
    dest_directory = FLAGS.model_dir
    # 如果文件夹不存在创建文件夹
    ensure_dir_exists(dest_directory)

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    # 如果模型tar压缩包不存在，
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):  # 创建一个私有函数, 制作进度条
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        statinfo = os.stat(filepath)    # 检查文件路径下的文件信息
        logging.info('Successfully downloaded ' + filename + ' {} bytes.'.format(statinfo.st_size))
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category,
                           sess, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    """ 创建单个 bottleneck 层缓存文件 """
    logging.info('Creating bottleneck at ' + bottleneck_path)
    # 获取图片路径
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not gfile.Exists(image_path):
        logging.fatal("File does not exist {}".format(image_path))

    # 获取图片二进制数据
    image_data = gfile.FastGFile(image_path, mode='rb').read()
    try:
        # 推理出bottleneck层的值
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor,
                                                    decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category,
                             bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                             resized_input_tensor, bottleneck_tensor, architecture):
    """从已有的缓存文件中读取数据或者缓存文件不存在时创建

    :param sess:
    :param image_lists:
    :param label_name:
    :param index:
    :param image_dir:
    :param category:
    :param bottleneck_dir:
    :param jpeg_data_tensor:
    :param decoded_image_tensor:
    :param resized_input_tensor:
    :param bottleneck_tensor:
    :param architecture: 使用的模型架构 Inception_v3 or Mobilenet

    :return: 一个图像在bottleneck层输出值的Numpy数组。
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    # 检查缓存目录是否存在
    ensure_dir_exists(sub_dir_path)

    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        # 创建缓存文件
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir,
                               category, sess, jpeg_data_tensor, decoded_image_tensor,
                               resized_input_tensor, bottleneck_tensor)

    # 读取缓存文件
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    # 缓存文件里面数据是否有效标志位, True表示重新读取缓存文件
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True

    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir,
                               category, sess, jpeg_data_tensor, decoded_image_tensor,
                               resized_input_tensor, bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor,
                      decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture):
    """ 缓存训练、测试、验证的所有bottleneck输出文件

    :param sess:
    :param image_lists:
    :param image_dir:
    :param bottleneck_dir:
    :param jpeg_data_tensor:
    :param decoded_image_tensor:
    :param resized_input_tensor:
    :param bottleneck_tensor:
    :param architecture:
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                                         category, bottleneck_dir, jpeg_data_tensor,
                                         decoded_image_tensor, resized_input_tensor,
                                         bottleneck_tensor, architecture)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    logging.info(str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir,
                                  image_dir, jpeg_data_tensor, decoded_image_tensor,
                                  resized_input_tensor, bottleneck_tensor, architecture):
    """随机获取缓存的bottlenecks值

    :param sess:
    :param image_lists:
    :param how_many: 获取bottlenecks值的数量，-1表示获取全部
    :param category:
    :param bottleneck_dir:
    :param image_dir:
    :param jpeg_data_tensor:
    :param decoded_image_tensor:
    :param resized_input_tensor:
    :param bottleneck_tensor:
    :param architecture:

    :return: bottlenecks数组列表、对应的分类标签的one-hot编码列表、图片文件路径列表
    """
    # 获取类别数量
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # 随机获取一组长度为how_many的bottlenecks数组
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]  # 0表示第一类（cats），1第二类（dogs）
            # 获取图片路径
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
            filenames.append(image_name)
            # 获取bottleneck层输出数据
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category,
                                                  bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                                                  resized_input_tensor, bottleneck_tensor, architecture)
            bottlenecks.append(bottleneck)
            # 分类标签的one-hot编码，只有属于的类别的对应位置的值才为1，其余为0
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            ground_truths.append(ground_truth)
    else:
        # 获取所有bottlenecks
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
                filenames.append(image_name)

                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category,
                                                      bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                                                      resized_input_tensor, bottleneck_tensor, architecture)
                bottlenecks.append(bottleneck)

                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                ground_truths.append(ground_truth)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir,
                                     input_jpeg_tensor, distorted_image, resized_input_tensor,
                                     bottleneck_tensor):
    """图像经过旋转、平移后的bottleneck层输出值, 不缓存

    :param sess:
    :param image_lists:
    :param how_many: 图片数量
    :param category:
    :param image_dir:
    :param input_jpeg_tensor:
    :param distorted_image: 输出图像变换结果的操作
    :param resized_input_tensor:
    :param bottleneck_tensor:

    :return: 经过图像变换的bottlenecks数组列表、对应的分类标签的one-hot编码列表
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)

        if not gfile.Exists(image_path):
            logging.fatal('File does not exist {}'.format(image_path))
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()

        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)

        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
    """是否启用变换,0表示不变

    :param flip_left_right: 是否以0.5的概率水平随机翻转训练图像 , bool
    :param random_crop: 裁剪为原图像边长的整数百分比, 整型
    :param random_scale: 比例变换（缩放）的百分比， 整型
    :param random_brightness: 像素数值随机乘以整数范围内的一个数， 整型

    :return: bool值，是否应用任何变换
    """
    return (flip_left_right or random_crop != 0) or (random_scale != 0) or (random_brightness != 0)


def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness,
                          input_width, input_height, input_depth, input_mean, input_std):
    """创建应用特殊变换的操作

    裁剪是通过在全图的一个随机位置放置一个边界框来完成的。裁剪参数控制框相对于输入图像的大小。
    如果为零，则方框的大小与输入相同，不进行裁剪。如果该值为50%，那么裁剪框将是输入宽度和高度的一半。width - crop%
    裁剪是先把图像双线性变换到（1+random_crop）尺寸，然后再随机裁剪到原来的1，这样能保证输出hwc维度还是原来一样

    缩放很像裁剪，除了边界框总是居中，它的大小在给定的范围内随机变化。
    例如，如果缩放百分比为零，那么边界框的大小与输入相同，并且不应用缩放。如果它是50%，那么边界框将在半宽半高和全尺寸之间的随机范围内。

    :param flip_left_right: bool
    :param random_crop: 裁剪 整型百分比
    :param random_scale: 缩放 整型百分比
    :param random_brightness: 像素数值随机乘以整数范围内的一个数， 整型百分比
    :param input_width:
    :param input_height:
    :param input_depth:
    :param input_mean:
    :param input_std:

    :return: 变换输入层占位符和变换结果输出操作
    """
    jpeg_data = tf.compat.v1.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)  # 增加通道维度
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)  # hwc->bhwc

    # 裁剪和缩放
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)

    resize_scale_value = tf.compat.v1.random_uniform(margin_scale_value.get_shape(),
                                                     minval=1.0,
                                                     maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.compat.v1.image.resize_bilinear(decoded_image_4d,
                                                          precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.compat.v1.random_crop(precropped_image_3d,
                                             [input_height, input_width, input_depth])

    # 水平翻转
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image

    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.compat.v1.random_uniform(flipped_image.get_shape(),
                                                   minval=brightness_min,
                                                   maxval=brightness_max)
    brightness_image = tf.multiply(flipped_image, brightness_value)

    # 标准化，减均值除方差
    offset_image = tf.subtract(brightness_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """ 使用tensorboard进行可视化 """
    with tf.name_scope('summaries'):
        # 计算张量各维度元素的平均值。
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            # 标准差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
        tf.compat.v1.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
    """添加最终识别的全连接和softmax操作层

    :param class_count:
    :param final_tensor_name: 生成最终结果的节点名称
    :param bottleneck_tensor: bottleneck层输出的数据操作
    :param bottleneck_tensor_size:

    :return: 训练和交叉熵结果的张量，以及瓶颈输出作为最后一层输入和对应真实标签的张量。
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.compat.v1.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.compat.v1.placeholder(tf.float32,
                                                      [None, class_count],
                                                      name='GroundTruthInput')

    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # 生成的值服从均值为0，标准差为0.001的正态分布
            initial_value = tf.compat.v1.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.compat.v1.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.compat.v1.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        # 计算交叉熵损失，真实标签和预测标签
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        # 优化器SGD
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(FLAGS.learning_rate)
        # 训练步长，根据损失函数更新学习率，使损失函数最小
        train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """

    :param result_tensor:
    :param ground_truth_tensor:

    :return: Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 预测标签与真实标签进行比较
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction,
                                          tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                # 计算准确率
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
    """保存模型"""
    # 用相同值的常量替换图中的所有变量。
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name]
    )
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def prepare_file_system():
    if gfile.Exists(FLAGS.summaries_dir):
        # 递归删除目录下的所有内容
        gfile.DeleteRecursively(FLAGS.summaries_dir)
    gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graph_dir)


def create_model_info(architecture):
    """给定模型的名称，返回关于它的信息

    使用基础模型进行迁移学习再训练，该函数进行模型名称到其中的一些属性的转换

    :param architecture:
    :return: 关于模型信息的字典，如果不能识别名称，则为None

    Raises:
        ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()  # 返回转换为小写的字符串副本。
    if architecture == 'inception_v3':
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299  # 299
        input_height = 299  # 299
        input_depth = 3  # 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128

    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            logging.error("Could't understand architecture name '%s'" % architecture)
            return None

        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                version_string != '0.5' and version_string != '0.25'):
            logging.error("The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',\
                          but found '%s' for architecture '%s'" % (version_string, architecture))
            return None

        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                size_string != '160' and size_string != '128'):
            logging.error("The Mobilenet input size should be '224', '192', '160', or '128',\
                          but found '%s' for architecture '%s'" % (size_string, architecture))
            return None

        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                logging.error("Couldn't understand architecture suffix '%s' for '%s'" % (parts[3], architecture))
                return None
            is_quantized = True
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
        data_url += version_string + '_' + size_string + '_frozen.tqz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        if is_quantized:
            model_base_name = 'quantized_graph.pd'
        else:
            model_base_name = 'frozen_graph.pd'
        model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        logging.error("Couldn't understand architecture name '%s'" % architecture)
        raise ValueError('Unknown architecture')

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
    }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
    """添加JPEG解码和调整图像大小的操作
        标准化，处理后数据平均值为0，方差为1

    :param input_width: 调整后的宽度
    :param input_height: 调整后的高度
    :param input_depth: 调整后的维度
    :param input_mean: 图片数据的均值
    :param input_std: 图片数据的方差

    :return: 节点输入JPEG数据的占位符，以及输出数据的预处理步骤
    """
    jpeg_data = tf.compat.v1.placeholder(tf.string, name='DecodeJPGInput')
    # 将JPEG编码的图像解码为uint8张量，
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, axis=0)  # 升维，hwc->bhwc
    resized_shape = tf.stack([input_height, input_width])
    resized_shape_as_int = tf.cast(resized_shape, dtype=tf.int32)
    # 使用双线性插值调整images为size.
    resized_image = tf.compat.v1.image.resize_bilinear(decoded_image_4d, resized_shape_as_int)
    # 减掉均值
    offset_image = tf.subtract(resized_image, input_mean)
    # 除以方差
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def main(_):
    logging.basicConfig(level=logging.DEBUG)  # 设置打印日志级别并输入到指定文件
    # logging.basicConfig(filename='example.log', level=logging.DEBUG, filemode='w')
    # critical(FATAL = CRITICAL)> error > warning > info > debug > NOTSET

    # 初始化文件夹
    prepare_file_system()
    ensure_dir_exists(FLAGS.summaries_dir)
    ensure_dir_exists(FLAGS.bottleneck_dir)
    ensure_dir_exists(os.path.split(FLAGS.output_graph)[0])

    # 获取模型信息
    model_info = create_model_info(FLAGS.architecture)
    if not model_info:
        logging.error('Did not recognize architecture flag')
        return -1

    # 建立预训练graph
    maybe_download_and_extract(model_info['data_url'])
    graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info)  # 返回的是模型对图像的操作

    # 创建数据集列表（存储的是文件名称）../MNIST  ../CatDog/Data/train
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_sets_percentage,
                                     FLAGS.validation_sets_percentage)
    for key, data in image_lists.items():
        for s in ['training', 'validation', 'testing']:
            logging.info("{}_{}: {}".format(key, s, len(data[s])))

    class_count = len(image_lists.keys())
    if class_count == 0:
        logging.error("No valid folders of images found at " + FLAGS.image_dir)
        return -1
    if class_count == 1:
        logging.error("Only one valid folder of images found at " + FLAGS.image_dir +
                      " - multiple classes are needed for classification.")
        return -1

    # 是否应用图像变换
    do_distort_images = should_distort_images(FLAGS.flip_left_right, FLAGS.random_crop,
                                              FLAGS.random_scale, FLAGS.random_brightness)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 动态申请显存
    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        # 建立图像解码子图。
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(  # 返回的是图像预处理的操作
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        if do_distort_images:
            # 应用变换操作
            logging.info("Apply transform op, don't used cache.")
            distorted_jpeg_data_tensor, distorted_image_tensor \
                = add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop,
                                        FLAGS.random_scale, FLAGS.random_brightness,
                                        model_info['input_width'], model_info['input_height'],
                                        model_info['input_depth'], model_info['input_mean'],
                                        model_info['input_std'])
        else:
            # 计算bottleneck层并缓存
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.architecture)

        # 添加要训练的最后的新层
        train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor \
            = add_final_training_ops(len(image_lists.keys()), FLAGS.final_tensor_name,
                                     bottleneck_tensor, model_info['bottleneck_tensor_size'])

        # 创建评估操作
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

        # 合并所有summaries，并且写入文件中，后通过tensorboard查看
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

        variable_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/validation')

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # 训练模型
        for i in range(FLAGS.epoch):
            # 获取一批输入bottleneck值，这些值要么是应用变换后重新计算得到，要么从存储在磁盘上的缓存得到
            if do_distort_images:
                train_bottlenecks, train_ground_truth \
                    = get_random_distorted_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training',
                                                       FLAGS.image_dir, distorted_jpeg_data_tensor,
                                                       distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                train_bottlenecks, train_ground_truth, _ \
                    = get_random_cached_bottlenecks(sess, image_lists, FLAGS.train_batch_size, 'training',
                                                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                                                    decoded_image_tensor, resized_image_tensor,
                                                    bottleneck_tensor, FLAGS.architecture)

            # 将bottleneck和实际类别输入到图中，并运行一个训练步骤。使用' merged '操作获取TensorBoard的训练摘要。
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # 每隔一段时间，打印出图表的训练情况。
            is_last_step = (i + 1 == FLAGS.epoch)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:   # 每过10个迭代打印一次训练结果
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={
                        bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth})
                logging.info("%s: Step %d: Train accuracy = %.1f%%" % (datetime.now(), i, train_accuracy*100))
                logging.info("%s: Step %d: Cross entropy = %f" % (datetime.now(), i, cross_entropy_value))

                variable_bottlenecks, variable_ground_truth, _ \
                    = get_random_cached_bottlenecks(sess, image_lists, FLAGS.validation_batch_size, 'validation',
                                                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                                                    decoded_image_tensor, resized_image_tensor,
                                                    bottleneck_tensor, FLAGS.architecture)

                # 每10个迭代验证一次，并用merged操作获取TensorBoard的训练摘要
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: variable_bottlenecks,
                               ground_truth_input: variable_ground_truth})
                variable_writer.add_summary(validation_summary, i)
                logging.info("%s: Step %d: Validation accuracy = %.1f%% (N=%d)" %
                             (datetime.now(), i, validation_accuracy*100, len(variable_bottlenecks)))

            # 存储中间结果, 经过多少个迭代进行存储，0为不存储
            intermediate_frequency = FLAGS.intermediate_store_frequency
            if intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0:
                intermediate_file_name = FLAGS.intermediate_output_graph_dir + 'intermediate_' + str(i) + '.pb'
                logging.info('Save intermediate result to: ' + intermediate_file_name)
                save_graph_to_file(sess, graph, intermediate_file_name)

        # 训练完成后，对一些之前没有使用过的新图像进行最后的测试评估。
        test_bottlenecks, test_ground_truth, test_filenames \
            = get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size, 'testing', FLAGS.bottleneck_dir,
                                            FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor,
                                            resized_image_tensor, bottleneck_tensor, FLAGS.architecture)
        test_accuracy, predictions = sess.run(
            [evaluation_step, prediction],
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        if FLAGS.print_misclassified_test_images:  # 输出测试集中分类错误的图像
            logging.info('=== MISCLASSIFED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    logging.info("%70s  %s" % (test_filename, list(image_lists.keys())[predictions[i]]))

        # 存储训练好的模型和类别文件
        save_graph_to_file(sess, sess.graph, FLAGS.output_graph)
        with gfile.GFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='./tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graph_dir',
        type=str,
        default='./tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="How many steps to store intermediate graph. If '0' then will not store."
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='./tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='./tmp/summary_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=1000,
        help='How many epochs to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_sets_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_sets_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a epoch.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""How many images to test on. This test set is only used once, to evaluate the final accuracy of the model
             after training completes.\nA value of -1 causes the entire test set to be used, which leads to more stable
             results across runs."""
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=-1,
        help="""How many images to use in an evaluation batch. This validation set is used much more often than test 
             set, and is an early indicator of how accurate the model is during training.\n
             A value of -1 causes the entire validation set to be used, which leads to more stable results across 
             training iterations, but may be slower on large training sets."""
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="Whether to print out a list of all misclassified test images.",
        action='store_true'     # 参数输入‘--print_misclassified_test_images’即为Ture（后面不能配置具体的值），不输入即为默认
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model',
        help="Path to classify_image_graph_def.pb, imagenet_synset_to_human_label_map.txt, and \
             imagenet_2012_challenge_label_map_proto.pbtxt"
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='./tmp/bottleneck',
        help="Path to cache bottleneck layer values as files."
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="The name of the output classification layer in the retrained graph."
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="Whether to randomly flip of the training images horizontally with a 1 in 2 chance.",
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="A percentage determining how much of a margin to randomly crop off the training images.\nA value of 0 is \
             not cropping."
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="A percentage determining how much to randomly scale up the size of the training images by.\nA value of 0 \
             is not scaling."
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="A percentage determining how much to randomly multiply the training image input pixels up or down by.\
             \nA value of 0 is not op."
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='inception_v3',
        help="""
                Which model architecture to use. 'inception_v3' is the most accurate, but
                also the slowest. For faster or smaller models, chose a MobileNet with the
                form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
                'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
                pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
                less accurate, but smaller and faster network that's 920 KB on disk and
                takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
                for more information on Mobilenet.
             """
    )

    # namespace, args
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
