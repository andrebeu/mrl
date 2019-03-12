from glob import glob as glob
import os,sys

import tensorflow as tf
import numpy as np
from mrl import *

# model specs
stsize = int(sys.argv[1])
optimizer = str(sys.argv[2])
lr = int(sys.argv[3]) 
gamma = int(sys.argv[4])
seed = int(sys.argv[5])

# train time
nsess = 10
epochs_per_sess = 1500
eval_nepochs = 1000

print('lr',lr/1e5)
opt_dict = {'rms':tf.train.RMSPropOptimizer(lr/1e5),
						'adam':tf.train.AdamOptimizer(lr/1e5)}

agent = MRLAgent(gamma=gamma/100,optimizer=opt_dict[optimizer],seed=seed)

## make dir
model_name = 'state_%i-gamma_%i-opt_%s-lr_%ie-5'%(stsize,gamma,optimizer,lr)
model_dir = 'models/sweep2/%s/%.3i'%(model_name,seed) 
os.makedirs(model_dir)

train_lossL = []
for sess in range(nsess):
	print('sess',sess/nsess)
	# train
	train_lossL.append(agent.train_curr(epochs_per_sess,eps=1))
	# eval
	eval_reward82 = agent.eval(eval_nepochs,np.array([.8,.2]))
	eval_reward28 = agent.eval(eval_nepochs,np.array([.2,.8]))

	# save model and eval data
	fpath = model_name + '-trepochs%i'%((sess+1)*epochs_per_sess)
	np.save(model_dir+'/'+fpath+'-evalrewards82',eval_reward82)
	np.save(model_dir+'/'+fpath+'-evalrewards28',eval_reward28)
	agent.saver_op.save(agent.sess,model_dir+'/'+fpath+'-model_chkpoint')
	# periodically save eval data
	
# save train loss
train_data = np.concatenate(train_lossL)
np.save(model_dir+'/'+model_name+'-train_loss',train_data)
