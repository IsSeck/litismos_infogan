import numpy as np
import tensorflow as tf
from ops import *
import h5py as hdf
import time 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import os
import shutil

# plot10
def plot10(samples):
    fig = plt.figure(figsize = (15,15))
    gs = gridspec.GridSpec(10, 13)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)

    return fig


###################### récupération du modèle ##########################
export_dir = "dccgan_model"

sess = tf.Session()
tf.saved_model.loader.load(sess,["dccgan_model"],export_dir)

# affichage des variables du modèle
#trainable_var = tf.trainable_variables()
trainable_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
for var in trainable_var:
	print(var.name)

######################## restoration des valeur ########################
saver = tf.train.Saver()
saver.restore(sess,tf.train.latest_checkpoint(".") )
########################################################################
# chargement de la base d'entraînement
f = hdf.File( "restricted_celebA.hdf5", 'r')
all_images = f["celebA_cropped64_im"][:]
all_attrib = f["celebA_attributes"][:]
f.close()
########################################################################


########################### initialisation des variables en attendant le fichier de config     ########################
n_attribute = 3 # restricted_celebA, binary attributes
z_dim = 100
batch_size = 128
im_shape = (batch_size,64,64,3)


########################################################################


graph = tf.get_default_graph()
y_ph = graph.get_tensor_by_name("y_ph:0")
z_ph = graph.get_tensor_by_name("z_ph:0")
image_ph = graph.get_tensor_by_name("image_ph:0")
d_loss = graph.get_tensor_by_name("d_loss:0")
g_loss = graph.get_tensor_by_name("g_loss:0")
generated_image = graph.get_tensor_by_name("generated_image:0")
d_solver = graph.get_operation_by_name("d_AdamOpt")
g_solver = graph.get_operation_by_name("g_AdamOpt") 

n_examples = 19200*3
n_forward_per_epoch = n_examples//batch_size
n_epoch = 35 # environ 3epoch/h 

export_dir = "dccgan_model/"
if os.path.exists(export_dir):
	shutil.rmtree(export_dir)

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
builder.add_meta_graph_and_variables(sess, ["dccgan_model"])

saver = tf.train.Saver(max_to_keep=5)

output_rep = "out/"
if not os.path.exists(output_rep):
    os.makedirs(output_rep)


# reprise de l'optimisation
for n in range(n_epoch):
	permut = np.random.permutation(19200*3)
	
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
	# après chaque demi-epoque je sauvegarde
	
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
####################################################################################################################################################
