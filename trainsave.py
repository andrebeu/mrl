from glob import glob as glob
import os,sys

import tensorflow as tf
import numpy as np
from mrl import *

# model specs
stsize = sys.argv[1]
optimizer = sys.argv[2]
lr = sys.argv[3] # divide by 1e-05
gamma = sys.argv[4]
# train time
nsess = 2 
epochs_per_sess = 100
eval_nepochs = 10

print('lr',lr/100000)
opt_dict = {'rms':tf.train.RMSPropOptimizer(lr/100000),
						'adam':tf.train.AdamOptimizer(lr/100000)}

agent = MRLAgent(gamma=gamma/100,optimizer=opt_dict[optimizer])

## make dir
model_name = 'state_%i-opt_%s-lr_%i'%(stsize,optimizer,lr)
num_models = len(glob('models/sweep1/%s/*'%model_name)) 
model_dir = 'models/sweep1/%s/%.3i'%(model_name,num_models) 
os.makedirs(model_dir)

for sess in range(nsess):
	print('sess',sess/nsess)
	fpath = model_name + '-trepochs%i'%((sess+1)*epochs_per_sess)
	# train
	agent.train(epochs_per_sess)
	# save model
	agent.saver_op.save(agent.sess,model_dir+'/'+fpath+'trained_model')
	# eval
	eval_reward = agent.eval(eval_nepochs)
	# periodically save eval data
	np.save(model_dir+'/'+fpath+'-evalrewards',eval_reward)
