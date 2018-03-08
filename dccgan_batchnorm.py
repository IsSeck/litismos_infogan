
#coding: utf-8



import numpy as np
import tensorflow as tf
from ops import *
import h5py as hdf
#import ipdb
import time 
import matplotlib.pyplot as plt # probleme d'installation de tkinter
import matplotlib.gridspec as gridspec
import os
import configparser
import shutil # pour la suppression de fichier non vide

# configuration du parser
cfg = configparser.ConfigParser(interpolation=configparser.BasicInterpolation())
cfg.read("dccgan.cfg")


# mise en place d'un générateur avec fractionnaly-strided conv #################################
section = "TRAINING_INIT_VAR"
n_attribute = cfg.getint(section,"n_attribute") # restricted_celebA, binary attributes
z_dim = cfg.getint(section,"z_dim")
batch_size = cfg.getint(section,"batch_size")
ih = cfg.getint(section,"im_height")
iw = cfg.getint(section,"im_width")
im_channel = cfg.getint(section,"im_channel")
im_shape = (batch_size,ih,iw,im_channel)
y_ph =  tf.placeholder(dtype = tf.float32, shape = (batch_size,n_attribute),name= "y_ph") 
z_ph = tf.placeholder(dtype = tf.float32, shape = (batch_size,z_dim), name = "z_ph"  )
image_ph = tf.placeholder( dtype = tf.float32, shape = im_shape, name = "image_ph"  )


# h0
section = "GENERATOR_SETUP"
g_height1 = cfg.getint(section,"g_height1")  # hauteur et largeur après projection et reshape 
g_num_channel1 = cfg.getint(section, "g_num_channel1") # nombre de canaux
n_out1 = g_height1*g_height1* g_num_channel1
n_in1 = n_attribute + z_dim

s_step = 2
#strides = [1,s_step,s_step,1] # le stride qui va être utilisé pour toutes
                    # les convolutions à pas fractionné

strides2 = [1,2,2,1]
strides1 = [1,1,1,1]

# h1
input_channel_conv1 = g_num_channel1 + n_attribute
output_channel_conv1 = cfg.getint(section,"output_channel_conv1") 
output_shape_h1 = [batch_size, s_step*g_height1,s_step*g_height1,output_channel_conv1]

# h2
input_channel_conv2 = output_channel_conv1 + n_attribute
output_channel_conv2 = cfg.getint(section,"output_channel_conv2")
output_shape_h2 = [batch_size, g_height1*s_step**2,g_height1*s_step**2,output_channel_conv2]

# h3
input_channel_conv3 = output_channel_conv2 + n_attribute
output_channel_conv3 = cfg.getint(section,"output_channel_conv3")
output_shape_h3 = [batch_size, g_height1*s_step**3,g_height1*s_step**3,output_channel_conv3]

# h4
input_channel_conv4 = output_channel_conv3 + n_attribute
output_channel_conv4 = cfg.getint(section,"output_channel_conv4")
output_shape_h4 = [batch_size, g_height1*s_step**4,g_height1*s_step**4,output_channel_conv4]

# mettre en place une couche h5 qui est une deconvolution avec un stride de 1
input_channel_conv5 = output_channel_conv4 + n_attribute
output_channel_conv5 = cfg.getint(section,"output_channel_conv5")
output_shape_h5 = [batch_size, g_height1*s_step**4,g_height1*s_step**4,output_channel_conv5]

# generator's variables
g_w1_std = cfg.getfloat(section,"g_w1_std")
g_b1_std = cfg.getfloat(section,"g_b1_std")
g_w1 = tf.Variable( tf.random_normal([n_out1, n_in1 ], stddev= g_w1_std),  "g_w1")
g_b1 = tf.Variable(tf.random_normal([n_out1,1], stddev = g_b1_std), "g_b1")


std_conv1 = cfg.getfloat(section, "std_conv1")
std_conv2 = cfg.getfloat(section, "std_conv2")
std_conv3 = cfg.getfloat(section, "std_conv3")
std_conv4 = cfg.getfloat(section, "std_conv4")
std_conv5 = cfg.getfloat(section, "std_conv5")
ks1 = cfg.getint(section,"kernel_size_conv1")
ks2 = cfg.getint(section,"kernel_size_conv2")
ks3 = cfg.getint(section,"kernel_size_conv3")
ks4 = cfg.getint(section,"kernel_size_conv4")
ks5 = cfg.getint(section,"kernel_size_conv5")
g_conv1_filters = tf.Variable(tf.random_normal([ks1,ks1,output_channel_conv1, input_channel_conv1], stddev = std_conv1), "g_conv1_filters")
g_conv2_filters = tf.Variable(tf.random_normal([ks2,ks2,output_channel_conv2, input_channel_conv2], stddev = std_conv2), "g_conv2_filters")
g_conv3_filters = tf.Variable(tf.random_normal([ks3,ks3,output_channel_conv3, input_channel_conv3], stddev = std_conv3), "g_conv3_filters")
g_conv4_filters = tf.Variable(tf.random_normal([ks4,ks4,output_channel_conv4, input_channel_conv4], stddev = std_conv4), "g_conv4_filters")
g_conv5_filters = tf.Variable(tf.random_normal([ks5,ks5,output_channel_conv5, input_channel_conv5], stddev = std_conv5), "g_conv5_filters")



def generator(z, y):  
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE) as scope:
        z_ = tf.concat([z, y],1, 'z_cond_concat')
        yb = tf.reshape(y,(batch_size,1,1,n_attribute ))
        #print "shape: {}".format( sess.run(z_).shape)
        intermediate_projected= tf.matmul(g_w1, z_, transpose_b = True) + g_b1
        projected = tf.transpose(intermediate_projected)
        
        h0 = tf.reshape(projected, (batch_size, g_height1,g_height1,g_num_channel1),"from_projected_to_h0" )
        h0 = tf.layers.batch_normalization(h0)
        h0 = lrelu(h0)
        h0 = conv_cond_concat(h0,yb)
        
        h0 = tf.layers.batch_normalization(h0)
        h1 = tf.nn.conv2d_transpose(h0,g_conv1_filters,output_shape_h1, strides2 )
        h1 = lrelu(h1)
        h1 = conv_cond_concat(h1,yb)
        
        h1 = tf.layers.batch_normalization(h1)
        h2 = tf.nn.conv2d_transpose(h1,g_conv2_filters, output_shape_h2, strides2 )
        h2 = lrelu(h2)
        h2 = conv_cond_concat(h2,yb)
        
        h2 = tf.layers.batch_normalization(h2)
        h3 = tf.nn.conv2d_transpose(h2,g_conv3_filters, output_shape_h3, strides2 )
        h3 = lrelu(h3)
        h3 = conv_cond_concat(h3,yb)
        
        h3 = tf.layers.batch_normalization(h3)
        h4 = tf.nn.conv2d_transpose(h3,g_conv4_filters, output_shape_h4, strides2 )
        h4 = lrelu(h4)
        h4 = conv_cond_concat(h4,yb)
        
        h5 = tf.nn.conv2d_transpose(h4,g_conv5_filters, output_shape_h5, strides1 )
    
        
        
    return tf.nn.sigmoid(h5)

################### fin mise en place générateur ##################################


################### code pour générer la sortie de la 4e couche de convolution ####
def h3_output_size(discriminateur_input_size, kernel_sizes,strides ):
    """
    inputs:
        + discriminateur_input_size: taille de l'entrée du discriminateur en hauteur(ou largeur)
        + kernel_size: vecteurs contenant les hauteurs (ou largeurs) des filtres
        + strides : vecteur contenant les strides
    output :
        + out: la hauteur (ou largeur) de la sortie de la L-e couche de convolution
    """
    if (len(kernel_sizes)!=len(strides)):
        print("Le 2e et le 3e arguments n'ont pas la même longueur")
        return 0
    else:
        L = len(kernel_sizes)
        out = discriminateur_input_size
        for i in range(L):
            out = np.ceil((out-kernel_sizes[i]+1)/strides[i])
        return int(out)
#                      

################# mise en place du discriminateur ###############################################

#stddev = 0.025
section = "DISCRIMINATOR_SETUP"
# nombre de filtres par couches de convolution
nfc1 = cfg.getint(section,"nfc1") 
nfc2 = cfg.getint(section,"nfc1") 
nfc3 = cfg.getint(section,"nfc1") 
nfc4 = cfg.getint(section,"nfc1") 

# kernel_size, hauteur(ou largeur des filtres)
ks1 = cfg.getint(section,"ks1")
ks2 = cfg.getint(section,"ks2")
ks3 = cfg.getint(section,"ks3")
ks4 = cfg.getint(section,"ks4")

# strides des convolutions
s1 = cfg.getint(section, "strides1"); stride1 = [1,s1,s1,1]
s2 = cfg.getint(section, "strides2"); stride2 = [1,s2,s2,1]
s3 = cfg.getint(section, "strides3"); stride3 = [1,s3,s3,1]
s4 = cfg.getint(section, "strides4"); stride4 = [1,s4,s4,1]
s_fc = h3_output_size(ih,[ks1,ks2,ks3,ks4],[s1,s2,s3,s4])

# std des différentes 
std1 = cfg.getfloat(section, "std1")
std2 = cfg.getfloat(section, "std2")
std3 = cfg.getfloat(section, "std3")
std4 = cfg.getfloat(section, "std4")
std_fc = cfg.getfloat(section, "std_fc")
std_fc_b = cfg.getfloat(section, "std_fc_b")

d_conv1 = tf.Variable(tf.random_normal([ks1,ks1,im_channel+n_attribute, nfc1], stddev=std1)) # 3 nombre de canaux de l'image d'entrées
d_conv2 = tf.Variable(tf.random_normal([ks2,ks2,nfc1+n_attribute, nfc2], stddev=std2))
d_conv3 = tf.Variable(tf.random_normal([ks3,ks3,nfc2+n_attribute, nfc3], stddev=std3))
d_conv4 = tf.Variable(tf.random_normal([ks4,ks4,nfc3+n_attribute, nfc4], stddev=std4))
d_w_fc = tf.Variable(tf.random_normal([s_fc*s_fc*nfc4+n_attribute, 1], stddev=std_fc))
d_b_fc = tf.Variable(tf.random_normal([1], stddev=std_fc_b))




def discriminator(image,y): # on suppose que les images sont de taille 64*64*3
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
        yb = tf.reshape(y, [batch_size,1,1,n_attribute])
        x = conv_cond_concat(image,yb)
        
        h0 = lrelu(tf.nn.conv2d(x, d_conv1, stride1, "VALID"))
        h0 =conv_cond_concat(h0,yb)
        
        h0 = tf.layers.batch_normalization(h0)
        h1 = lrelu( tf.nn.conv2d(h0,d_conv2,stride2, "VALID") )
        h1 = conv_cond_concat(h1,yb)
        
        h1= tf.layers.batch_normalization(h1)
        h2 = lrelu( tf.nn.conv2d(h1,d_conv3,stride3, "VALID") )
        h2 = conv_cond_concat(h2,yb)

        h2 = tf.layers.batch_normalization(h2)
        h3 = lrelu(tf.nn.conv2d(h2,d_conv4,stride4, "VALID") )
        h3 = tf.reshape(h3,[batch_size,-1])
        h3 = tf.concat([h3,y],1)

        
        h3 = tf.layers.batch_normalization(h3)
        h3 = tf.reshape(h3,[batch_size, -1])

		
        
        output_logit = tf.matmul(h3,d_w_fc) + d_b_fc

        return tf.nn.sigmoid(output_logit), output_logit

####################### fin mise en place du discriminateur #############################


 
########################## génération d'image, et calcul des loss ######################
generated_image = tf.identity(generator(z_ph,y_ph), "generated_image")
d_real, d_logit_real = discriminator(image_ph,y_ph)
d_fake, d_logit_fake = discriminator(generated_image,y_ph)


d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logit_real, labels = tf.ones_like(d_logit_real))) 
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logit_fake, labels = tf.zeros_like(d_logit_fake)))
d_loss = tf.add(d_loss_real, d_loss_fake, "d_loss")
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logit_fake,labels = tf.ones_like(d_logit_fake)), name = "g_loss")

theta_d = [d_conv1, d_conv2, d_conv3,d_w_fc,d_b_fc]
theta_g = [g_w1,g_b1,g_conv1_filters,g_conv2_filters,g_conv3_filters,g_conv4_filters]

section = "TRAINING_INIT_VAR"
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_learning_rate = cfg.getfloat(section, "g_learning_rate")
d_learning_rate = cfg.getfloat(section, "d_learning_rate")
g_beta1 = cfg.getfloat(section, "g_beta1")
d_beta1 = cfg.getfloat(section, "d_beta1")

with tf.control_dependencies(update_ops):
	d_solver = tf.train.AdamOptimizer(learning_rate = d_learning_rate, beta1 = d_beta1, name = "d_AdamOpt").minimize(d_loss, var_list = theta_d )
	g_solver = tf.train.AdamOptimizer(learning_rate = g_learning_rate, beta1 = g_beta1, name = "g_AdamOpt").minimize(g_loss, var_list = theta_g )

############################### fin des calculs de fonction de cout ####################

################################ mise en place des summary #######################################



# plot10
def plot10(samples):
    fig = plt.figure(figsize = (15,15))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)

    return fig



#####################################################################################################



# In[13]:
#########################  petit code de test du generateur et du discriminateur #####################


export_dir = cfg.get("FOLDERS","export_dir")

if os.path.exists(export_dir):
	shutil.rmtree(export_dir)

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

sess =  tf.Session()


n_examples = cfg.getint("TRAINING_INIT_VAR","n_examples")
n_forward_per_epoch = n_examples//batch_size
n_epoch = cfg.getint("TRAINING_INIT_VAR","n_epoch") 

s = time.time()
dataset_file = cfg.get("TRAINING_DATA","database_name")
f = hdf.File( dataset_file, 'r')
all_images = f["celebA_cropped64_im"][:]
all_attrib = f["celebA_attributes"][:]
f.close()

saver = tf.train.Saver(max_to_keep=5)

output_rep = cfg.get("FOLDERS","output_rep")
if not os.path.exists(output_rep):
    os.makedirs(output_rep)

init = tf.global_variables_initializer()
sess.run(init)

tag = cfg.get("FOLDERS", "tag")
builder.add_meta_graph_and_variables(sess, [tag])
j=0

for n in range(n_epoch):
	permut = np.random.permutation(n_examples)
	
	all_images = all_images[permut, :]
	all_attrib = all_attrib[permut, :]
	
	
	# après avoir permuté les images, et les labels, on les met dans des batch qu'on va permuter

	batch_indices_matrix = np.random.randint(1,size=(n_forward_per_epoch,batch_size))
	for i in range(n_forward_per_epoch):
		batch_indices_matrix[i,:] = range(i*batch_size,(i+1)*batch_size)
	permut = np.random.permutation(n_forward_per_epoch)
	batch_indices_matrix = batch_indices_matrix[permut,:]

	
	s = time.time()
	for i in range(n_forward_per_epoch//2):
		#print("iteration:{}".format(i))
		batch_indices = batch_indices_matrix[i,:]
		im_batch = np.reshape(all_images[batch_indices,:]/255.0,(batch_size,64,64,3)) # create var im_height = 64 and im_chan = 3
		attri_batch = all_attrib[batch_indices,:]
		
		
		
		#print("im_batch, type : {}, shape : {}".format(type(im_batch),im_batch.shape))
		z_batch = np.random.uniform(low=-1.0, high=1.0, size=(batch_size,z_dim))
		sess.run(d_solver,feed_dict={image_ph:im_batch, z_ph:z_batch , y_ph : attri_batch })
		# train generator twice
		z_batch = np.random.uniform(low=-1.0, high=1.0, size=(batch_size,z_dim))
		sess.run(g_solver , feed_dict = { z_ph:z_batch, y_ph : attri_batch }  )

		#z_batch = np.random.uniform(low=-1.0, high=1.0, size=(batch_size,z_dim))
		#sess.run(g_solver, feed_dict = { z_ph:z_batch, y_ph : attri_batch }  )
	
	builder.save()
	saver.save(sess, "./my_conv_generator.ckpt", global_step =2* n)
	print (" epoque {}, temps mis {}, g_loss: {} ".format(n,time.time()-s, sess.run(g_loss, feed_dict={ z_ph:z_batch, y_ph : attri_batch } )))
	
	samples = sess.run(generated_image, feed_dict={z_ph: z_batch, y_ph: attri_batch})
	
	fig = plot10(samples[:64,:,:,:])
	plt.savefig(output_rep + "{}.png".format(str(2*n).zfill(4)), bbox_inches='tight')
	plt.close(fig)
	# sauvegarde après chaque demi-epoque 
	
	s = time.time()
	for i in range(n_forward_per_epoch//2,n_forward_per_epoch):
		batch_indices = batch_indices_matrix[i,:]
		im_batch = np.reshape(all_images[batch_indices,:]/255.0,(batch_size,64,64,3)) # create var im_height = 128 and im_chan = 3
		attri_batch = all_attrib[batch_indices,:]
		z_batch = np.random.uniform(low=-1.0, high=1.0, size=(batch_size,z_dim))
		sess.run(d_solver,feed_dict={image_ph:im_batch, z_ph:z_batch , y_ph : attri_batch })
		# train generator twice
		z_batch = np.random.uniform(low=-1.0, high=1.0, size=(batch_size,z_dim))
		sess.run(g_solver , feed_dict = { z_ph:z_batch, y_ph :attri_batch  }  )
		#z_batch = np.random.uniform(low=-1.0, high=1.0, size=(batch_size,z_dim))
		#sess.run(g_solver, feed_dict = { z_ph:z_batch, y_ph : attri_batch  }  )
		
	#print "iter {} temps mis {} batch_size {} ".format(i,time.time()-s,batch_size)
	saver.save(sess, "./my_conv_generator.ckpt", global_step = 2*n+1)
	print (" epoque {}, temps mis {}, g_loss: {} ".format(n,time.time()-s, sess.run(g_loss, feed_dict={ z_ph:z_batch, y_ph : attri_batch } )))
	samples = sess.run(generated_image, feed_dict={z_ph: z_batch, y_ph: attri_batch})
	builder.save()



	fig = plot10(samples[:64,:,:,:])
	plt.savefig(output_rep + "{}.png".format(str(2*n+1).zfill(4)), bbox_inches='tight')
	plt.close(fig)
sess.close()
###############################################################################
