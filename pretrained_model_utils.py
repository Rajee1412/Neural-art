
# coding: utf-8

# In[208]:

# reference http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
import os
import sys
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np

get_ipython().magic('matplotlib inline')


# In[209]:

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 300
CHANNELS = 3
VGG_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
NOISE_RATIO = 0.6

#print (VGG_MEANS)


# In[219]:

def load_pretrained_VGG():
    model_path = 'imagenet-vgg-verydeep-19.mat'
    model = scipy.io.loadmat(model_path)

    layers = model['layers']

    def _weights (layer, expected_layer_name):
        WB = layers[0][layer][0][0][2]
        weight = WB[0][0]
        bias = WB[0][1]
        layer_name = layers[0][layer][0][0][0][0]
        assert expected_layer_name == layer_name

        return weight, bias

    def _conv_2d(prev_layer,layer, layer_name):

        W, B = _weights(layer,layer_name)
        w = tf.constant(W)
        b = tf.constant(np.reshape(B,(B.size)))

        print (b.shape)

        conv_layer = tf.nn.conv2d(prev_layer,filter=w,strides=[1,1,1,1],padding='SAME')+b
        return conv_layer

    def _conv2d_relu(prev_layer,layer, layer_name):

        conv_layer = _conv_2d(prev_layer,layer,layer_name)

        return tf.nn.relu(conv_layer)

    def _avgpool(prev_layer):

        return tf.nn.avg_pool(prev_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])

    return graph



# In[211]:

def get_noise_image(content_image):

    noise_image = np.random.uniform(-20,20,(1,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS)).astype('float32')

    out_image =  noise_image*NOISE_RATIO + content_image*(1-NOISE_RATIO)

    return noise_image





# In[212]:

def resize_image(image,name):
#     return tf.image.resize_image_with_crop_or_pad(image,IMAGE_HEIGHT,IMAGE_WIDTH)
    try:
        im = Image.open(image)
        resized = im.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
        resized.save('images/resized'+'_'+str(name)+'.jpg')
    except IOError:
        print ("cannot create thumbnail for '%s'" % resized)




# In[213]:

def reshape_image(image):
    reshaped_image = np.reshape(image,((1,)+image.shape))

    reshaped_norm_image = reshaped_image - VGG_MEANS

    return reshaped_norm_image


# ### Test method

# In[214]:

resize_image('images/tvar_2.jpg','tvar_2')

resized_img = scipy.misc.imread('images/resized_tvar_2.jpg')


#plt.imshow(resized_img)

reshaped_image = reshape_image(resized_img)

#print (reshaped_image.shape)

#print (reshaped_image[0])

noise_image = get_noise_image(reshaped_image)

#print (noise_image[0])


# In[215]:

def save_image(path,image):
    #print (image.shape)
    denorm_image = sum(image,VGG_MEANS)

    image = np.clip(denorm_image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


# In[220]:




# In[ ]:
