# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-12-28 16:16:49
# @Last Modified by:   ashayaan
# @Last Modified time: 2020-02-23 23:03:36

import os, time
import itertools, imageio, pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
import numpy as np
from numpy import array

sizePixel1 = 512
sizePixel2 = sizePixel1*sizePixel1

size_d = 32

batch_size = 4
train_epoch = 100

folder = '/home/shayaan/Columbia/sem_1/convolutional_gans/Data2/png'
pngLocation = 'Results_3'
root = 'outputConditionalGANs/'
model = 'DCGAN_conditional_run3_'
code_dir = '/home/shayaan/Columbia/sem_1/convolutional_gans'

print("--------------------------")
print(code_dir)
print("--------------------------")

output_dir = code_dir + root

if not os.path.isdir(output_dir):
	os.makedirs(output_dir)

def read_tensor_from_image_file(path, input_height=sizePixel1, input_width=sizePixel1, input_mean=0, input_std=255):
	
	input_name = "file_reader"
	output_name = "normalized"
	file_reader = tf.read_file(path, input_name)
	image_reader = tf.image.decode_png(file_reader, channels = 1)
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0);
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.Session()
	result = sess.run(normalized)
	sess.close()
	return result


# In[4]:


nSeries = 222
numInSeries = 11
nImages = nSeries*numInSeries #2442
img  = []

counter = 0
for filename in os.listdir(folder):
	print (filename)
	counter += 1
	print (counter)
	image = read_tensor_from_image_file(folder+'/'+filename)
	number_of_blobs = filename.split('_')[0]
	label = np.full((512,512,1),number_of_blobs)
	image = image.reshape(512,512,1)
	img.append(np.concatenate((image,label),axis=2))
	if counter == 1:
		break


# for j in range(0,numInSeries):
#     for i in range(1,nSeries+1):
#         print(counter,i,j)
#         fname = str(i) + '_' + str(j) + '.png'
#         path = folder + '/' + fname
#         orig_img = read_tensor_from_image_file(path)

#         img[counter] = orig_img.reshape(sizePixel1,sizePixel1)
#         counter = counter+1


def lrelu(x, th=0.2):
	return tf.maximum(th * x, x)



def compute_score_np(imgSample):
	
	tmp1 = imgSample.copy()
	#tmp1 = tmp1.reshape(512,512)
	#tmp2 = tmp1
	min1 = tmp1.min()
	max1 = tmp1.max()
	
	tmp1[tmp1< 0.5]=0.0
	tmp1[tmp1>=0.5]=1.0 
	#print(min1)
	#print(max1)
	tmp1 = 1.0-tmp1
	label_im, nb_labels = scipy.ndimage.label(tmp1)
	
	#print(nb_labels)
	
	#plt.subplot(121)
	#plt.imshow(tmp1)
	
	#plt.subplot(122)
	#plt.imshow(tmp2)

	return float(nb_labels), min1, max1


# In[ ]:



	


# In[8]:


def generator(x, isTrain=True, reuse=False):
	print('*******************')
	with tf.variable_scope('generator', reuse=reuse):
		print(x.shape)
		conv1 = tf.layers.conv2d_transpose(x, size_d*2*2*2*2*2*2, [2, 2], strides=(1, 1), padding='valid')
		lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
		print(lrelu1.shape)
		# 2nd hidden layer
		conv2 = tf.layers.conv2d_transpose(lrelu1, size_d*2*2*2*2, [7, 7], strides=(4, 4), padding='same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
		print(lrelu2.shape)
		# 3rd hidden layer
		conv3 = tf.layers.conv2d_transpose(lrelu2, size_d*2*2, [7, 7], strides=(4, 4), padding='same')
		lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
		print(lrelu3.shape)
		# 4th hidden layer
		conv4 = tf.layers.conv2d_transpose(lrelu3, size_d, [7, 7], strides=(4, 4), padding='same')
		lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
		print(lrelu4.shape)
		conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [7, 7], strides=(4, 4), padding='same')
		o = tf.nn.sigmoid(conv5)
		print ("*****************")
		print (lrelu4.shape)
		print (o)
		
		return o


def discriminator(x, isTrain=True, reuse=False):
	with tf.variable_scope('discriminator', reuse=reuse):
		
		print('')
		print('inside discriminator')
		print('====================')
		print('x')
		print(x.get_shape())
		
		# 1st hidden layer
		conv1 = tf.layers.conv2d(x, size_d, [7, 7], strides=(4, 4), padding='same')
		lrelu1 = lrelu(conv1, 0.2)
		
		# 2nd hidden layer
		conv2 = tf.layers.conv2d(lrelu1, size_d*2*2, [7, 7], strides=(4, 4), padding='same')
		lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

		# 3rd hidden layer
		conv3 = tf.layers.conv2d(lrelu2, size_d*2*2*2*2, [7, 7], strides=(4, 4), padding='same')
		lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
		
		# 4th hidden layer
		conv4 = tf.layers.conv2d(lrelu3, size_d*2*2*2*2*2*2, [7, 7], strides=(4, 4), padding='same')
		lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

		conv5 = tf.layers.conv2d(lrelu4, 1, [2, 2], strides=(1, 1), padding='valid')
		o = tf.nn.sigmoid(conv5)

		return o, conv5


def plot_sample(samples, size1, size2):
	
	fig1 = plt.figure(figsize=(size1, size2))
	gs = gridspec.GridSpec(size1, size2)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(512, 512), cmap='gray')

	return fig1


fixed_z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))


def show_result(fixed_z_, num_epoch, show = False, save = False, path = 'result.png'):
	
	
	test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
	print ("Printing the shape of test_images")
	print (test_images.shape)

	size_figure_grid = 2
	fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize = (5, 5))
	for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
		ax[i, j].get_xaxis().set_visible(False)
		ax[i, j].get_yaxis().set_visible(False)

	for k in range(size_figure_grid*size_figure_grid):
		i = k // size_figure_grid
		j = k % size_figure_grid
		ax[i, j].cla()
		ax[i, j].imshow(np.reshape(test_images[k], (512, 512)), cmap='gray')

	label = 'Epoch {0}'.format(num_epoch)
	fig.text(0.5, 0.04, label, ha='center')

	if save:
		plt.savefig(path)

	if show:
		plt.show()
	else:
		plt.close()


# In[13]:


def show_train_hist(hist, show = False, save = False, path = 'Train_hist_con.png'):
	x = range(len(hist['D_losses']))

	y1 = hist['D_losses']
	y2 = hist['G_losses']

	plt.plot(x, y1, label='D_loss')
	plt.plot(x, y2, label='G_loss')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	plt.legend(loc=4)
	plt.grid(True)
	plt.tight_layout()

	if save:
		plt.savefig(path)

	if show:
		plt.show()
	else:
		plt.close()


# In[14]:


def next_batch(data, num):
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = array([data[ i] for i in idx])
	return data_shuffle

lr = 0.0002

x = tf.placeholder(tf.float32, shape=(None, 512, 512, 2))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 200))

isTrain = tf.placeholder(dtype=tf.bool)
lda = tf.placeholder(tf.float32)

g_z = tf.placeholder(tf.float32, shape=(512, 512, 1))
G_z = generator(z, isTrain)

#Conditioning the image generated by the generator on the
#number of labels
G_z_ = tf.concat((G_z,x[:,:,:,-1:]),3)

D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z_, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake


# In[17]:


scores = []
minValues = []
maxValues = []


# In[18]:


scores_1 = tf.placeholder(tf.float32, [None, 1])
mins_1 = tf.placeholder(tf.float32, [None, 1])
maxs_1 = tf.placeholder(tf.float32, [None, 1])

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))
scores_G = tf.reduce_mean(scores_1)
mins_G = tf.reduce_mean(mins_1)
maxs_G = tf.reduce_mean(maxs_1)
G_loss = G_loss + lda*scores_G

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
	G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_set = np.array(img).reshape(len(img), sizePixel1, sizePixel1, 2)
print('shape before')
print(train_set.shape)
#train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1
print('shape after')
print(train_set.shape)
# results save folder

if not os.path.isdir(root):
	os.mkdir(root)
if not os.path.isdir(root + pngLocation):
	os.mkdir(root + pngLocation)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


# In[ ]:


x_ = next_batch(train_set, batch_size)

# print('shape')
# print(x_.shape)


# In[16]:


# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

max_iter = 400
for epoch in range(train_epoch):
	G_losses = []
	G_scores = []
	G_mins = []
	G_maxs = []
	D_losses = []
	epoch_start_time = time.time()
	
	if epoch%2 == 0:
		lda_ = 1.0
	else:
		lda_ = 1.0
	
	for iter in range(max_iter):

		# update discriminator
		x_ = next_batch(train_set, batch_size)
		z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

		label = x_[:,0,0,1]
		label = label.reshape(label.shape[0],1,1,1)
		label = np.repeat(label,100,axis=3)
		z_ = np.concatenate((z_,label),axis=3)
		# print ("Shape of the generator input")
		# print (z_.shape)
		loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True, lda: lda_})
		D_losses.append(loss_d_)

		# update generator
		z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
		z_ = np.concatenate((z_,label),axis=3)   
		# generate a batch of images
		G_z_samples = sess.run(G_z, feed_dict={z: z_, isTrain: True})
   
		# np functions to compute score
		scores = []
		minValues = []
		maxValues = []
		for i in range(batch_size):

			score, minValue, maxValue = compute_score_np(G_z_samples[i])

			scores.append(score)
			minValues.append(minValue)
			maxValues.append(maxValue)

		scores = np.expand_dims(np.array(scores), axis=1)
		minValues = np.expand_dims(np.array(minValues), axis=1)
		maxValues = np.expand_dims(np.array(maxValues), axis=1)
		fd = {x: x_, z: z_, isTrain: True, lda: lda_, scores_1: scores, mins_1: minValues, maxs_1: maxValues}
		loss_g_, scores_g_, mins_g_, maxs_g_, _ = sess.run([G_loss, scores_G, mins_G, maxs_G, G_optim], feed_dict=fd)
		G_losses.append(loss_g_)
		G_scores.append(scores_g_)
		G_mins.append(mins_g_)
		G_maxs.append(maxs_g_)

	x_ = next_batch(train_set, batch_size)
	label = x_[:,0,0,1]
	label = label.reshape(label.shape[0],1,1,1)
	label = np.repeat(label,100,axis=3)
	fixed_z = np.concatenate((fixed_z_,label),axis=3)

	epoch_end_time = time.time()
	per_epoch_ptime = epoch_end_time - epoch_start_time
	print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f, scores: %.3f, mins: %.3f, maxs: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses), np.mean(G_scores), np.mean(G_mins), np.mean(G_maxs)))
	fixed_p = root + pngLocation + '/' + model + str(epoch + 1) + '.png'
	show_result(fixed_z,(epoch + 1), save=True, path=fixed_p)
	train_hist['D_losses'].append(np.mean(D_losses))
	train_hist['G_losses'].append(np.mean(G_losses))
	train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
	pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
	img_name = root + pngLocation + '/' + model + str(e + 1) + '.png'
	images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()
