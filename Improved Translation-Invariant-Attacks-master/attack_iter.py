"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
start_time = time.clock()

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import skimage
import tensorflow as tf
import cv2
import csv

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate
from tensorpack.tfutils.tower import TowerContext
slim = tf.contrib.slim


tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string('checkpoint_path_inception_v3', '../models/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_v4', '../models/inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_resnet_v2', '../models/inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet', '../models/resnet_v2_152.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_vgg', '../models/vgg_16.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', '../models/adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', '../models/ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', '../models/ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', '../models/ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_dir', '../../data/images', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', '../../data_out/images', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 32.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 35, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0 , 'Momentum.')

tf.flags.DEFINE_float('prob', 0.4, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('sacle_times', 5, 'the times of scales.')

FLAGS = tf.flags.FLAGS

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

kernel = gkern(15, 2).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath,'rb') as f:
            image = imresize(imread(f, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def load_label_class(input_dir):
  """Loads target classes."""
  #with tf.gfile.Open(os.path.join(input_dir, 'dev.csv')) as f:
  with tf.gfile.Open('../../data/dev.csv') as f:
  
    return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2} 



def save_images(images_adv, filenames, output_dir,eps,images):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            
            image_gs=cv2.GaussianBlur(images_adv[i,:,:,:],(7,7),-0.1)
            image_gsc=np.clip(image_gs,images[i,:,:,:]-eps,images[i,:,:,:]+eps)
            imsave(f, (image_gsc+ 1.0) * 0.5, format='png')
            imsave(FLAGS.output_dir+'_ngs/'+filename, (images_adv[i,:,:,:]+ 1.0) * 0.5, format='png')

def target_model(x):

    num_classes=1001
  
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            input_diversity(x), num_classes=num_classes, is_training=False)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            input_diversity(x), num_classes=num_classes, is_training=False, reuse=True)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet, end_points_resnet = resnet_v2.resnet_v2_152(
            input_diversity(x), num_classes=num_classes, is_training=False)


    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
    

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
    
  
    logits = (logits_v3 +1.6*logits_adv_v3+logits_v4 +logits_res_v2 + logits_resnet+logits_ens3_adv_v3+logits_ens4_adv_v3+logits_ensadv_res_v2) / 8.6
    auxlogits = ( 1.6*end_points_adv_v3['AuxLogits'] +end_points_v3['AuxLogits']+end_points_ens3_adv_v3['AuxLogits']
     + end_points_v4['AuxLogits']+end_points_res_v2['AuxLogits']+end_points_ens4_adv_v3['AuxLogits']+end_points_ensadv_res_v2['AuxLogits']) / 7.6

  
    return logits,auxlogits



def scale_noise(x,y,x0,i,noise_sum):
    i+=1
    logits,auxlogits=target_model(x/tf.pow(2.0,i))
    cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)
    linf_loss=tf.norm(x-x0,np.inf)
    loss=cross_entropy+0.3*linf_loss
    noise= tf.gradients(loss, x)[0]
    noise_sum+=noise

    return x,y,x0,i,noise_sum

def cond_scale(x,y,x0,i,noise_sum):
    return tf.less(i,FLAGS.sacle_times)

def graph(x, y, i, x_max, x_min, grad,x0):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum

    k=tf.constant(0.0,tf.float32)
    noise_sum=tf.zeros(shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
    _,_,_,_,noise_sum=tf.while_loop(cond_scale,scale_noise,[x,y,x0,k,noise_sum])


    logits,auxlogits=target_model(x)
    cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)
    linf_loss=tf.norm(x-x0,np.inf)
    loss=cross_entropy+0.3*linf_loss
    noise= tf.gradients(loss, x)[0]

    noise=(noise_sum+noise)/(FLAGS.sacle_times+1)

    noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
  
    noise = momenstum*grad + noise

    x = x + alpha*tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise,x0


def stop(x, y, i, x_max, x_min, grad,x0):
    
    return tf.less(i,FLAGS.num_iter)


def input_diversity(input_tensor):
    
    #加入噪声和和其他diversity的方法

    input_tensor=tf.add(input_tensor,tf.random_uniform(shape=(FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3),
        minval=-0.5,maxval=0.5))
 
    input_tensor=tf.image.random_flip_left_right(input_tensor)
    input_tensor=tf.image.random_brightness(input_tensor,0.5)

    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)



def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    all_images_label_class = load_label_class(FLAGS.input_dir)
    print(time.clock() - start_time)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x0_input = tf.placeholder(tf.float32, shape=batch_shape)
        inf_w = tf.placeholder(tf.float32, shape=())
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)
        y = tf.one_hot(predicted_labels, num_classes)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _,_= tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad,x0_input])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
   

        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s2.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
            s3.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
            s4.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
            s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
            s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
            s7.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
            s8.restore(sess, FLAGS.checkpoint_path_resnet)
         
            print(time.clock() - start_time)
           
            for it in [340]:
                k=0
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                   
                    # print(label_class_for_batch.shape())
                    start_time_a_loop=time.clock()
                    adv_images = sess.run(x_adv, feed_dict={x_input: images, x0_input:images})
                    save_images(adv_images, filenames, FLAGS.output_dir+str(it),eps,images)
                    k+=1
                    print("the "+str(k)+" loop time used:",time.clock() - start_time_a_loop)
                print(time.clock() - start_time)


if __name__ == '__main__':
    tf.app.run()
