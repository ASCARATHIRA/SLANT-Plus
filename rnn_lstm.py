import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import utility
import sys
import time
# Global config variables

start_time = time.time()
print >> sys.stderr, 'Start time', start_time 
N = int(sys.argv[1])
folder = ''
if len(sys.argv)>2:
    folder = sys.argv[2]
    if not folder[-1]=='/':
        folder = folder + '/'
# THRESHOLD = 0.1
hidden_layer_size = N
input_size = N
target_size = 1

""" 
Initializations
"""

inv_map = utility.get_id_map(folder+'good_id_to_old_id.json',N)
adj = utility.get_adj(folder+'edgelist.txt', inv_map, N)
H = utility.get_history(folder+'opinion.txt', inv_map, N)
if len(H)>7005:
	H = H[:7000]
G = utility.make_adj_H(H,N)


X = utility.get_m_H(H)
a,b,c = utility.make_input(H[0],H[0],N,G)
user=[a]
dt=[b]
dm=[c]

for i in range(1, len(H)):
    a,b,c=utility.make_input(H[i],H[i-1],N,G)
    user.append(a)
    dt.append(b)
    dm.append(c)


total_size = len(H)

pre_processed = time.time()
print >> sys.stderr, 'After pre-processing', pre_processed 
batch_size = int(0.9*(len(H)))
num_classes = 2
state_size = N
learning_rate = 0.1


"""
Placeholders
"""
U = tf.placeholder(tf.float32, [total_size,state_size], name='user')
T = tf.placeholder(tf.float32, [total_size,state_size], name='delta_t')
M = tf.placeholder(tf.float32, [total_size,state_size], name='delta_m')
O = tf.placeholder(tf.float32, [total_size], name='sentiments')
ADJ = tf.placeholder(tf.float32, [state_size,state_size], name='adj_')
init_state = tf.zeros([state_size])


"""
Weights to be trained by the network
"""
with tf.variable_scope('rnn'):
    w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    with tf.variable_scope('input'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('output'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('forget'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('memory'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
  
with tf.variable_scope('rnn_op'):
    w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    with tf.variable_scope('input'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('output'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('forget'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('memory'):
        w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))  

def rnn_cell_gate(u,t,m,state,aFlag): 
    w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    temp = (m - w5)
    sig1 = w3 * tf.sigmoid(w4 * temp)
    sig2 = w6 * tf.sigmoid(-w7 * (m - w8))
    if aFlag==1 :
        h = tf.sigmoid(w1 * tf.exp(-w2 * t) * state + tf.matmul([u], W)[0,:] * tf.matmul([u], ADJ)[0,:] * (sig1 - sig2))
    else :
        h = tf.tanh(w1 * tf.exp(-w2 * t) * state + tf.matmul([u], W)[0,:] * tf.matmul([u], ADJ)[0,:] * (sig1 - sig2))
    return h

def rnn_cell(u,t,m,state,c_state,state_x,c_state_x): 
    with tf.variable_scope('rnn', reuse=True):
       	w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
        w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        with tf.variable_scope('input', reuse=True):
            inp_gate = rnn_cell_gate(u,t,m,state,1)
        with tf.variable_scope('forget', reuse=True):
            for_gate = rnn_cell_gate(u,t,m,state,1)
        with tf.variable_scope('output', reuse=True):
            out_gate = rnn_cell_gate(u,t,m,state,1)
        with tf.variable_scope('memory', reuse=True):
            new_mem = rnn_cell_gate(u,t,m,state,0)    
        c = (for_gate*c_state) + (inp_gate*new_mem)
        h = out_gate*tf.tanh(c)
        lamb = tf.exp(w9 + w10 * t + w11 * h)

    with tf.variable_scope('rnn_op', reuse=True):
        w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
        w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        with tf.variable_scope('input', reuse=True):
            inp_gate_x = rnn_cell_gate(u,t,m,state,1)
        with tf.variable_scope('forget', reuse=True):
            for_gate_x = rnn_cell_gate(u,t,m,state,1)
        with tf.variable_scope('output', reuse=True):
            out_gate_x = rnn_cell_gate(u,t,m,state,1)
        with tf.variable_scope('memory', reuse=True):
            new_mem_x = rnn_cell_gate(u,t,m,state,0)    
        c_x = (for_gate_x*c_state_x) + (inp_gate_x*new_mem_x)
        h_x = out_gate_x*tf.tanh(c_x)
        x = tf.tanh(w9 + w11 * h_x)

    return c,h,lamb,c_x,h_x,x


state = init_state
c_state = init_state
state_x = init_state
c_state_x = init_state
lambdas = []
lamb_c_states = []
lamb_states = []
opinions = []
op_states = []
op_c_states = []

for i in range(batch_size):
    c_state,state,lamb,c_state_x,state_x,lamb_x = rnn_cell(tf.gather(U,i),tf.gather(T,i),tf.gather(M,i),state,c_state,state_x,c_state_x)
    lambdas.append(lamb)
    lamb_states.append(state)
    lamb_c_states.append(c_state)

    opinions.append(lamb_x)
    op_states.append(state_x)
    op_c_states.append(c_state_x)


def get_integral(lamb_states):
    ret = 0.0
    with tf.variable_scope('rnn', reuse=True):
        w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
        w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
        for i in range(batch_size-1):
            ret += tf.reduce_sum((tf.exp(w9+w10*tf.gather(T,i+1)+w11*lamb_states[i]) - tf.exp(w9+w11*lamb_states[i]))/w10)
    return ret

def log_lambda(lambdas):
    log_sum=0.0
    for i in range(batch_size):
        log_sum += tf.log(tf.reduce_sum(lambdas[i]*tf.gather(U,i)))
    return log_sum

def get_mse(opinions):
    val = 0.0
    for i in range(batch_size):
        val+= tf.reduce_sum(tf.pow(tf.reduce_sum(opinions[i]*tf.gather(U,i)) - tf.gather(O,i), 2))
    return val

total_loss = tf.Variable(tf.zeros([], dtype=tf.float32), name='total_loss')
losses = get_integral(lamb_states) - log_lambda(lambdas) + get_mse(opinions)
total_loss = losses

with tf.variable_scope('rnn',reuse=True):
    t_w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    t_w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    t_w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    with tf.variable_scope('input',reuse=True):
        t_i_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_i_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('output'):
        t_o_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_o_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('forget'):
        t_f_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_f_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('memory'):
        t_c_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_c_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
  
with tf.variable_scope('rnn_op',reuse=True):
    t_x_w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    t_x_w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    t_x_w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    with tf.variable_scope('input',reuse=True):
        t_xi_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xi_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('output',reuse=True):
        t_xo_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xo_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('forget',reuse=True):
        t_xf_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xf_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    with tf.variable_scope('memory',reuse=True):
        t_xc_w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
        t_xc_W= tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))  

tf.scalar_summary('total_loss', tf.reshape(total_loss,[]))
merged = tf.merge_all_summaries()
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
saver = tf.train.Saver()

T_w9 = np.zeros((state_size),dtype=np.float32)
T_w10 = np.zeros((state_size),dtype=np.float32)
T_w11 = np.zeros((state_size),dtype=np.float32)

T_i_w1 = np.zeros((state_size),dtype=np.float32)
T_i_w2 = np.zeros((state_size),dtype=np.float32)
T_i_w3 = np.zeros((state_size),dtype=np.float32)
T_i_w4 = np.zeros((state_size),dtype=np.float32)
T_i_w5 = np.zeros((state_size),dtype=np.float32)
T_i_w6 = np.zeros((state_size),dtype=np.float32)
T_i_w7 = np.zeros((state_size),dtype=np.float32)
T_i_w8 = np.zeros((state_size),dtype=np.float32)
T_i_W = np.zeros((state_size,state_size),dtype=np.float32)

T_o_w1 = np.zeros((state_size),dtype=np.float32)
T_o_w2 = np.zeros((state_size),dtype=np.float32)
T_o_w3 = np.zeros((state_size),dtype=np.float32)
T_o_w4 = np.zeros((state_size),dtype=np.float32)
T_o_w5 = np.zeros((state_size),dtype=np.float32)
T_o_w6 = np.zeros((state_size),dtype=np.float32)
T_o_w7 = np.zeros((state_size),dtype=np.float32)
T_o_w8 = np.zeros((state_size),dtype=np.float32)
T_o_W = np.zeros((state_size,state_size),dtype=np.float32)

T_f_w1 = np.zeros((state_size),dtype=np.float32)
T_f_w2 = np.zeros((state_size),dtype=np.float32)
T_f_w3 = np.zeros((state_size),dtype=np.float32)
T_f_w4 = np.zeros((state_size),dtype=np.float32)
T_f_w5 = np.zeros((state_size),dtype=np.float32)
T_f_w6 = np.zeros((state_size),dtype=np.float32)
T_f_w7 = np.zeros((state_size),dtype=np.float32)
T_f_w8 = np.zeros((state_size),dtype=np.float32)
T_f_W = np.zeros((state_size,state_size),dtype=np.float32)

T_c_w1 = np.zeros((state_size),dtype=np.float32)
T_c_w2 = np.zeros((state_size),dtype=np.float32)
T_c_w3 = np.zeros((state_size),dtype=np.float32)
T_c_w4 = np.zeros((state_size),dtype=np.float32)
T_c_w5 = np.zeros((state_size),dtype=np.float32)
T_c_w6 = np.zeros((state_size),dtype=np.float32)
T_c_w7 = np.zeros((state_size),dtype=np.float32)
T_c_w8 = np.zeros((state_size),dtype=np.float32)
T_c_W = np.zeros((state_size,state_size),dtype=np.float32)

T_x_w9 = np.zeros((state_size),dtype=np.float32)
T_x_w10 = np.zeros((state_size),dtype=np.float32)
T_x_w11 = np.zeros((state_size),dtype=np.float32)

T_xi_w1 = np.zeros((state_size),dtype=np.float32)
T_xi_w2 = np.zeros((state_size),dtype=np.float32)
T_xi_w3 = np.zeros((state_size),dtype=np.float32)
T_xi_w4 = np.zeros((state_size),dtype=np.float32)
T_xi_w5 = np.zeros((state_size),dtype=np.float32)
T_xi_w6 = np.zeros((state_size),dtype=np.float32)
T_xi_w7 = np.zeros((state_size),dtype=np.float32)
T_xi_w8 = np.zeros((state_size),dtype=np.float32)
T_xi_W = np.zeros((state_size,state_size),dtype=np.float32)

T_xo_w1 = np.zeros((state_size),dtype=np.float32)
T_xo_w2 = np.zeros((state_size),dtype=np.float32)
T_xo_w3 = np.zeros((state_size),dtype=np.float32)
T_xo_w4 = np.zeros((state_size),dtype=np.float32)
T_xo_w5 = np.zeros((state_size),dtype=np.float32)
T_xo_w6 = np.zeros((state_size),dtype=np.float32)
T_xo_w7 = np.zeros((state_size),dtype=np.float32)
T_xo_w8 = np.zeros((state_size),dtype=np.float32)
T_xo_W = np.zeros((state_size,state_size),dtype=np.float32)

T_xf_w1 = np.zeros((state_size),dtype=np.float32)
T_xf_w2 = np.zeros((state_size),dtype=np.float32)
T_xf_w3 = np.zeros((state_size),dtype=np.float32)
T_xf_w4 = np.zeros((state_size),dtype=np.float32)
T_xf_w5 = np.zeros((state_size),dtype=np.float32)
T_xf_w6 = np.zeros((state_size),dtype=np.float32)
T_xf_w7 = np.zeros((state_size),dtype=np.float32)
T_xf_w8 = np.zeros((state_size),dtype=np.float32)
T_xf_W = np.zeros((state_size,state_size),dtype=np.float32)

T_xc_w1 = np.zeros((state_size),dtype=np.float32)
T_xc_w2 = np.zeros((state_size),dtype=np.float32)
T_xc_w3 = np.zeros((state_size),dtype=np.float32)
T_xc_w4 = np.zeros((state_size),dtype=np.float32)
T_xc_w5 = np.zeros((state_size),dtype=np.float32)
T_xc_w6 = np.zeros((state_size),dtype=np.float32)
T_xc_w7 = np.zeros((state_size),dtype=np.float32)
T_xc_w8 = np.zeros((state_size),dtype=np.float32)
T_xc_W = np.zeros((state_size,state_size),dtype=np.float32)

def train_network(num_epochs,state_size=N, verbose=True):
    with tf.Session() as sess:
        import os
        if not os.path.isdir(folder+'logs_lstm'+str(N)):
            os.mkdir(folder+'logs_lstm'+str(N))
        train_writer = tf.train.SummaryWriter(folder+'logs_lstm'+str(N), sess.graph)
        sess.run(tf.initialize_all_variables())
        training_losses = []
        if os.path.isfile(folder+"model_lstm"+str(N)+".ckpt"):
            training_state = np.zeros((state_size))
            saver.restore(sess, folder+"model_lstm"+str(N)+".ckpt")
            T_w9, T_w10,T_w11= sess.run(
                [t_w9,t_w10,t_w11],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_i_w1,T_i_w2,T_i_w3,T_i_w4,T_i_w5,T_i_w6,T_i_w7,T_i_w8,T_i_W= sess.run(
                [t_i_w1,t_i_w2,t_i_w3,t_i_w4,t_i_w5,t_i_w6,t_i_w7,t_i_w8,t_i_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_f_w1,T_f_w2,T_f_w3,T_f_w4,T_f_w5,T_f_w6,T_f_w7,T_f_w8,T_f_W= sess.run(
                [t_f_w1,t_f_w2,t_f_w3,t_f_w4,t_f_w5,t_f_w6,t_f_w7,t_f_w8,t_f_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_o_w1,T_o_w2,T_o_w3,T_o_w4,T_o_w5,T_o_w6,T_o_w7,T_o_w8,T_o_W = sess.run(
                [t_o_w1,t_o_w2,t_o_w3,t_o_w4,t_o_w5,t_o_w6,t_o_w7,t_o_w8,t_o_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_c_w1,T_c_w2,T_c_w3,T_c_w4,T_c_w5,T_c_w6,T_c_w7,T_c_w8,T_c_W = sess.run(
                [t_c_w1,t_c_w2,t_c_w3,t_c_w4,t_c_w5,t_c_w6,t_c_w7,t_c_w8,t_c_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_x_w9, T_x_w10,T_x_w11= sess.run(
                [t_x_w9,t_x_w10,t_x_w11],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xi_w1,T_xi_w2,T_xi_w3,T_xi_w4,T_xi_w5,T_xi_w6,T_xi_w7,T_xi_w8,T_xi_W= sess.run(
                [t_xi_w1,t_xi_w2,t_xi_w3,t_xi_w4,t_xi_w5,t_xi_w6,t_xi_w7,t_xi_w8,t_xi_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xf_w1,T_xf_w2,T_xf_w3,T_xf_w4,T_xf_w5,T_xf_w6,T_xf_w7,T_xf_w8,T_xf_W= sess.run(
                [t_xf_w1,t_xf_w2,t_xf_w3,t_xf_w4,t_xf_w5,t_xf_w6,t_xf_w7,t_xf_w8,t_xf_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xo_w1,T_xo_w2,T_xo_w3,T_xo_w4,T_xo_w5,T_xo_w6,T_xo_w7,T_xo_w8,T_xo_W = sess.run(
                [t_xo_w1,t_xo_w2,t_xo_w3,t_xo_w4,t_xo_w5,t_xo_w6,t_xo_w7,t_xo_w8,t_xo_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xc_w1,T_xc_w2,T_xc_w3,T_xc_w4,T_xc_w5,T_xc_w6,T_xc_w7,T_xc_w8,T_xc_W = sess.run(
                [t_xc_w1,t_xc_w2,t_xc_w3,t_xc_w4,t_xc_w5,t_xc_w6,t_xc_w7,t_xc_w8,t_xc_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
        else:
            epoch = 0
            training_loss = 100000
            while epoch < num_epochs:
                training_state = np.zeros((state_size))
                if verbose:
                    summary,training_loss_, _ = sess.run([merged,total_loss,train_step],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
                    training_loss = training_loss_
                    train_writer.add_summary(summary, epoch)
                    if epoch%10==0 :
                        if verbose:
                            print >> sys.stderr, "Average loss at EPOCH ",epoch,": ", training_loss
                            print "Average loss at EPOCH ",epoch,": ", training_loss
                            training_losses.append(training_loss)
                epoch+=1
            f = open('epoch'+str(N)+'.txt', 'w')
            print >>f, "Total epochs: ",epoch
            f.close()
            T_w9, T_w10,T_w11= sess.run(
                [t_w9,t_w10,t_w11],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_i_w1,T_i_w2,T_i_w3,T_i_w4,T_i_w5,T_i_w6,T_i_w7,T_i_w8,T_i_W= sess.run(
                [t_i_w1,t_i_w2,t_i_w3,t_i_w4,t_i_w5,t_i_w6,t_i_w7,t_i_w8,t_i_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_f_w1,T_f_w2,T_f_w3,T_f_w4,T_f_w5,T_f_w6,T_f_w7,T_f_w8,T_f_W= sess.run(
                [t_f_w1,t_f_w2,t_f_w3,t_f_w4,t_f_w5,t_f_w6,t_f_w7,t_f_w8,t_f_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_o_w1,T_o_w2,T_o_w3,T_o_w4,T_o_w5,T_o_w6,T_o_w7,T_o_w8,T_o_W = sess.run(
                [t_o_w1,t_o_w2,t_o_w3,t_o_w4,t_o_w5,t_o_w6,t_o_w7,t_o_w8,t_o_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_c_w1,T_c_w2,T_c_w3,T_c_w4,T_c_w5,T_c_w6,T_c_w7,T_c_w8,T_c_W = sess.run(
                [t_c_w1,t_c_w2,t_c_w3,t_c_w4,t_c_w5,t_c_w6,t_c_w7,t_c_w8,t_c_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_x_w9, T_x_w10,T_x_w11= sess.run(
                [t_x_w9,t_x_w10,t_x_w11],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xi_w1,T_xi_w2,T_xi_w3,T_xi_w4,T_xi_w5,T_xi_w6,T_xi_w7,T_xi_w8,T_xi_W= sess.run(
                [t_xi_w1,t_xi_w2,t_xi_w3,t_xi_w4,t_xi_w5,t_xi_w6,t_xi_w7,t_xi_w8,t_xi_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xf_w1,T_xf_w2,T_xf_w3,T_xf_w4,T_xf_w5,T_xf_w6,T_xf_w7,T_xf_w8,T_xf_W= sess.run(
                [t_xf_w1,t_xf_w2,t_xf_w3,t_xf_w4,t_xf_w5,t_xf_w6,t_xf_w7,t_xf_w8,t_xf_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xo_w1,T_xo_w2,T_xo_w3,T_xo_w4,T_xo_w5,T_xo_w6,T_xo_w7,T_xo_w8,T_xo_W = sess.run(
                [t_xo_w1,t_xo_w2,t_xo_w3,t_xo_w4,t_xo_w5,t_xo_w6,t_xo_w7,t_xo_w8,t_xo_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            T_xc_w1,T_xc_w2,T_xc_w3,T_xc_w4,T_xc_w5,T_xc_w6,T_xc_w7,T_xc_w8,T_xc_W = sess.run(
                [t_xc_w1,t_xc_w2,t_xc_w3,t_xc_w4,t_xc_w5,t_xc_w6,t_xc_w7,t_xc_w8,t_xc_W],feed_dict={U:user,T:dt,M:dm,init_state:training_state,O:X,ADJ:adj})
            saver.save(sess, folder+ "model_lstm"+str(N)+".ckpt")
    return T_w9, T_w10, T_w11, T_i_w1, T_i_w2, T_i_w3, T_i_w4, T_i_w5, T_i_w6, T_i_w7, T_i_w8, T_i_W, T_o_w1, T_o_w2, T_o_w3, T_o_w4, T_o_w5, T_o_w6, T_o_w7, T_o_w8, T_o_W, T_f_w1, T_f_w2, T_f_w3, T_f_w4, T_f_w5, T_f_w6, T_f_w7, T_f_w8, T_f_W, T_c_w1, T_c_w2, T_c_w3, T_c_w4, T_c_w5, T_c_w6, T_c_w7, T_c_w8, T_c_W, T_x_w9,T_x_w10, T_x_w11, T_xi_w1, T_xi_w2, T_xi_w3, T_xi_w4, T_xi_w5, T_xi_w6, T_xi_w7, T_xi_w8, T_xi_W, T_xo_w1, T_xo_w2, T_xo_w3, T_xo_w4, T_xo_w5, T_xo_w6, T_xo_w7, T_xo_w8, T_xo_W, T_xf_w1, T_xf_w2, T_xf_w3, T_xf_w4, T_xf_w5, T_xf_w6, T_xf_w7, T_xf_w8, T_xf_W,  T_xc_w1, T_xc_w2, T_xc_w3, T_xc_w4, T_xc_w5, T_xc_w6, T_xc_w7, T_xc_w8, T_xc_W

T_w9, T_w10, T_w11, T_i_w1, T_i_w2, T_i_w3, T_i_w4, T_i_w5, T_i_w6, T_i_w7, T_i_w8, T_i_W, T_o_w1, T_o_w2, T_o_w3, T_o_w4, T_o_w5, T_o_w6, T_o_w7, T_o_w8, T_o_W, T_f_w1, T_f_w2, T_f_w3, T_f_w4, T_f_w5, T_f_w6, T_f_w7, T_f_w8, T_f_W, T_c_w1, T_c_w2, T_c_w3, T_c_w4, T_c_w5, T_c_w6, T_c_w7, T_c_w8, T_c_W, T_x_w9,T_x_w10, T_x_w11, T_xi_w1, T_xi_w2, T_xi_w3, T_xi_w4, T_xi_w5, T_xi_w6, T_xi_w7, T_xi_w8, T_xi_W, T_xo_w1, T_xo_w2, T_xo_w3, T_xo_w4, T_xo_w5, T_xo_w6, T_xo_w7, T_xo_w8, T_xo_W, T_xf_w1, T_xf_w2, T_xf_w3, T_xf_w4, T_xf_w5, T_xf_w6, T_xf_w7, T_xf_w8, T_xf_W,  T_xc_w1, T_xc_w2, T_xc_w3, T_xc_w4, T_xc_w5, T_xc_w6, T_xc_w7, T_xc_w8, T_xc_W = train_network(1000)
end_time = time.time()
print >> sys.stderr, 'Duration', end_time - start_time, 'seconds'

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def test_inp_gate(u,t,m,state):
    temp = (m - T_i_w5)
    sig1 = T_i_w3 * sigmoid(T_i_w4 * temp)
    sig2 = T_i_w6 * sigmoid(-T_i_w7 * (m - T_i_w8))
    h = sigmoid(T_i_w1 * np.exp(-T_i_w2 * t) * state + np.matmul([u], T_i_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def test_out_gate(u,t,m,state):
    temp = (m - T_o_w5)
    sig1 = T_o_w3 * sigmoid(T_o_w4 * temp)
    sig2 = T_o_w6 * sigmoid(-T_o_w7 * (m - T_o_w8))
    h = sigmoid(T_o_w1 * np.exp(-T_o_w2 * t) * state + np.matmul([u], T_o_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def test_for_gate(u,t,m,state):
    temp = (m - T_f_w5)
    sig1 = T_f_w3 * sigmoid(T_f_w4 * temp)
    sig2 = T_f_w6 * sigmoid(-T_f_w7 * (m - T_f_w8))
    h = sigmoid(T_f_w1 * np.exp(-T_f_w2 * t) * state + np.matmul([u], T_f_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def test_new_mem(u,t,m,state):
    temp = (m - T_c_w5)
    sig1 = T_c_w3 * sigmoid(T_c_w4 * temp)
    sig2 = T_c_w6 * sigmoid(-T_c_w7 * (m - T_c_w8))
    h = sigmoid(T_c_w1 * np.exp(-T_c_w2 * t) * state + np.matmul([u], T_c_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def test_x_inp_gate(u,t,m,state):
    temp = (m - T_xi_w5)
    sig1 = T_xi_w3 * sigmoid(T_xi_w4 * temp)
    sig2 = T_xi_w6 * sigmoid(-T_xi_w7 * (m - T_xi_w8))
    h = sigmoid(T_xi_w1 * np.exp(-T_xi_w2 * t) * state + np.matmul([u], T_xi_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def test_x_out_gate(u,t,m,state):
    temp = (m - T_xo_w5)
    sig1 = T_xo_w3 * sigmoid(T_xo_w4 * temp)
    sig2 = T_xo_w6 * sigmoid(-T_xo_w7 * (m - T_xo_w8))
    h = sigmoid(T_xo_w1 * np.exp(-T_xo_w2 * t) * state + np.matmul([u], T_xo_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def test_x_for_gate(u,t,m,state):
    temp = (m - T_xf_w5)
    sig1 = T_xf_w3 * sigmoid(T_xf_w4 * temp)
    sig2 = T_xf_w6 * sigmoid(-T_xf_w7 * (m - T_xf_w8))
    h = sigmoid(T_xf_w1 * np.exp(-T_xf_w2 * t) * state + np.matmul([u], T_xf_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def test_x_new_mem(u,t,m,state):
    temp = (m - T_xc_w5)
    sig1 = T_xc_w3 * sigmoid(T_xc_w4 * temp)
    sig2 = T_xc_w6 * sigmoid(-T_xc_w7 * (m - T_xc_w8))
    h = sigmoid(T_xc_w1 * np.exp(-T_xc_w2 * t) * state + np.matmul([u], T_xc_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    return h

def rnn_cell_test(u,t,m,state,c_state,state_x,c_state_x):
    c = (test_for_gate(u,t,m,state)* c_state) + (test_inp_gate(u,t,m,state) * test_new_mem(u,t,m,state))
    h = test_out_gate(u,t,m,state)*np.tanh(c)
    lamb = np.exp(T_w9 + T_w10 * t + T_w11 * h)

    c_x = (test_x_for_gate(u,t,m,state)* c_state_x) + (test_x_inp_gate(u,t,m,state) * test_x_new_mem(u,t,m,state))
    h_x = test_x_out_gate(u,t,m,state)*np.tanh(c_x)
    x = np.tanh(T_x_w9 + T_x_w11 * h_x)
    return c,h,lamb,c_x,h_x,x


def t_get_integral(t_lamb_states):
    ret = 0.0
    for i in range(batch_size,total_size-1):
        ret += np.add.reduce(np.exp(T_w9+T_w10*dt[i+1]+T_w11*t_lamb_states[i-batch_size]) - np.exp(T_w9+T_w11*t_lamb_states[i-batch_size]) / T_w10)
    return ret

def t_log_lambda(t_lambdas):
    log_sum=0.0
    for i in range(batch_size,total_size):
        log_sum += np.log(np.add.reduce(t_lambdas[i-batch_size]*user[i]))
    return log_sum

def t_get_mse(t_opinions):
    val = 0.0
    for i in range(batch_size,total_size):
        val+= np.power(np.add.reduce(t_opinions[i-batch_size]*user[i]) - X[i], 2)
    return val

def test_network():
    t_state = np.zeros((state_size))
    t_c_state = np.zeros((state_size))

    t_lambdas = []
    t_lamb_states = []
    t_lamb_c_states = []


    t_state_x = np.zeros((state_size))
    t_c_state_x = np.zeros((state_size))

    t_opinions = []
    t_op_states = []
    t_op_c_states = []

    for i in range(batch_size,total_size):
        t_c_state, t_state, t_lamb, t_c_state_x, t_state_x, t_x = rnn_cell_test(user[i],dt[i],dm[i],t_state,t_c_state,t_state_x,t_c_state_x)
        t_lambdas.append(t_lamb)
        t_lamb_states.append(t_state)
        t_lamb_c_states.append(t_c_state)

        t_opinions.append(t_x)
        t_op_states.append(t_state_x)
        t_op_c_states.append(t_c_state_x)

    mse_loss = t_get_mse(t_opinions)
    lamb_loss = t_get_integral(t_lamb_states) - t_log_lambda(t_lambdas)
    test_loss = mse_loss + lamb_loss
    return mse_loss,lamb_loss,test_loss

mse_loss_,lamb_loss_,test_loss_= test_network()
print >> sys.stderr, "(MSE, SE, LAMBDA, TOTAL LOSS)", mse_loss_/float(total_size-batch_size), mse_loss_,lamb_loss_,test_loss_
print "(MSE, SE, LAMBDA, TOTAL LOSS)", mse_loss_/float(total_size-batch_size), mse_loss_,lamb_loss_,test_loss_
end_time = time.time()
print >> sys.stderr, 'Duration', end_time - start_time, 'seconds'

