import numpy as np
import tensorflow as tf 
from tensorflow.contrib import rnn

with open('Trial2Buy04Pattern','r') as f:
    content = f.readlines()
	
# generate the sequence pattern of each uuid	
uuid_pattern_list = [x.split('\t')[2].strip('{''}').split(',') for x in content]  

#Transfer each event to an index number
def getIndex(uuid_pattern_list): 
    index = {}
    i = 1
    for pattern in uuid_pattern_list:
        for action in pattern:
            if index.get(action) == None:
                index[action] = i
                i+=1
    index["(PURCHASE)"] = i
    index["Padding"] = 0
    return index 
	
# input an index, return the corresponding event.	
def getKeyByValue(dic,value): 
    for key, _value in dic.items():
        if value == _value:
            return key
			
# generate an input-shape-form data w.r.t. index number.			
def pattern_input_shape (index_value): 
    data = np.zeros(shape = (len(index)))
    data[index_value] = 1
    return data

# get training or testing batch
def getBatch(source, batch_size, time_step, batch_cursor, step_cursor):
	batch_input = []
	batch_true = []
	_data = []
	for cursor in batch_cursor:
		temp = step_cursor[cursor] + time_step
		for _step in range(step_cursor[cursor], step_cursor[cursor]+time_step+1):
			try:
				_data.append(source[cursor][_step])
			except:
				if _step != temp:
					padding = np.asarray([0. for x in range(len(index))])
					padding [0] = 1
					_data.append(padding)
				else:
					_data.append(pattern_input_shape(index['(PURCHASE)']))
					step_cursor[cursor] = 0
					#print('Reseting step_cursor...')
			step_cursor[cursor]+=1
		batch_input.append([ _data[x] for x in range(0,time_step)])
		batch_true.append(_data[time_step])
		_data = []
		batch_cursor[batch_cursor.index(cursor)]+=1
        
		if batch_cursor[batch_cursor.index(cursor+1)] > len(source)-1:
			batch_cursor[batch_cursor.index(cursor+1)]=0
			print ("Reseting batch_cursor...")
	return batch_input, batch_true
		
index = getIndex(uuid_pattern_list)

# generate a list with format in inputshape from uuid_pattern_list. --> length = len(uuid_pattern_list)
uuid_pattern_list_input = [ [pattern_input_shape(index[action]) for action in each_pattern] for each_pattern in uuid_pattern_list]

# calculate average pattern length of each user and print out.
event_sum = 0
pattern_number=0
for pattern in uuid_pattern_list_input:
    pattern_number += 1
    event_sum += len(pattern)
print ('Average event pattern length for each user is (not including "PURCHASE") :', "{:.2f}".format(event_sum/pattern_number))

# seperate training:testing = 8:2

testing_source= uuid_pattern_list_input[0:int(len(uuid_pattern_list_input)*0.2)]
training_source= uuid_pattern_list_input[int(len(uuid_pattern_list_input)*0.2):]

# configuration of batch size and time step in each process 
batch_size = 30
time_step = 3

# configuration of cursor
training_step_cursor = [0 for i in range(len(training_source))]
segment = (len(training_source)//batch_size)
training_batch_cursor = [offset * segment for offset in range(batch_size)]

testing_step_cursor = [0 for i in range(len(testing_source))]
segment = (len(testing_source)//batch_size)
testing_batch_cursor = [offset * segment for offset in range(batch_size)]

# configuration of training arguments 
learning_rate = 0.001
training_iter = 31200
display_step = 10
n_hidden = 256
n_classes = len(index)


x = tf.placeholder('float', [None, time_step, len(index)])
y = tf.placeholder('float',[None, len(index)])

weights = {
    'out':tf.Variable(tf.random_normal([n_hidden, len(index)]))
}

biases = {
    'out':tf.Variable(tf.random_normal([len(index)]))
}

# define lstm model 
def lstmModel(x, weights, biases):
    x = tf.unstack(x, time_step, 1)
    
    #lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
    lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    #lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0).
    lstm_layers = rnn.MultiRNNCell([lstm_dropout]* 3)
    
    #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_hidden, state_is_tuple=True)
    #outputs, states = rnn.static_rnn (lstm_cell, x, dtype=tf.float32)rnn.static_rnn)
    outputs, states = tf.nn.dynamic_rnn(lstm_layers, x, dtype=tf.float32)
    
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = lstmModel(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# start
with tf.Session() as sess:
    sess.run(init)
    step = 1
 
    # training
    while step * batch_size < training_iter:
        batch_x, batch_y = getBatch(training_source, batch_size, time_step, training_batch_cursor, training_step_cursor)
        
        sess.run(optimizer, feed_dict= {x: batch_x , y: batch_y})
        if step % display_step ==0:
            acc = sess.run(accuracy, feed_dict= {x: batch_x , y: batch_y})
            
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    
    #tesing 
    
    testing_times = 0
    testing_accu = 0 
    for i in range(0, len(testing_source)//batch_size*3):
        test_data ,test_label = getBatch(testing_source, batch_size, time_step, testing_batch_cursor, testing_step_cursor)
        testing_accu += sess.run(accuracy, feed_dict={x: test_data, y: test_label})
        testing_times += 1
    print ('Average testing accuracy is :', "{:.2f}".format(testing_accu/testing_times))
    #Application 


