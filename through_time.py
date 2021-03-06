import numpy as np
import tensorflow as tf
import utility2
import sys
import time
import math

start_time = time.time()
print >> sys.stderr, 'Start time', start_time 
N = int(sys.argv[1])
folder = sys.argv[2]
model_file = sys.argv[3]

# THRESHOLD = 0.1
hidden_layer_size = N
input_size = N
target_size = 1

inv_map = utility2.get_id_map(folder+'good_id_to_old_id.json',N)
adj = utility2.get_adj(folder+'edgelist.txt', inv_map, N)
H = utility2.get_history(folder+'opinion.txt', inv_map, N)
G = utility2.make_adj_H(H,N)
'''
X = utility.make_opinions(folder+'new_opinion.txt',H,inv_map,N)
X = [X[i][:N] for i in range(len(H)) if int(H[i][0]) < N]
'''

X = utility2.get_m_H(H)
a,b,c = utility2.make_input(H[0],H[0],N,G)
user=[a]
dt=[b]
dm=[c]

for i in range(1, len(H)):
    a,b,c=utility2.make_input(H[i],H[i-1],N,G)
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
#O = tf.placeholder(tf.float32, [total_size,state_size],name='o')
init_state = tf.zeros([state_size])


"""
Function to train the network
"""
with tf.variable_scope('rnn_cell'):
    w1 = tf.get_variable('w1', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w2 = tf.get_variable('w2', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w3 = tf.get_variable('w3', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w4 = tf.get_variable('w4', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w5 = tf.get_variable('w5', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w6 = tf.get_variable('w6', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w7 = tf.get_variable('w7', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w8 = tf.get_variable('w8', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w9 = tf.get_variable('w9', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w10 = tf.get_variable('w10', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    w11 = tf.get_variable('w11', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    W = tf.get_variable('W', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

    w1_x = tf.get_variable('w1_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w2_x = tf.get_variable('w2_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w3_x = tf.get_variable('w3_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w4_x = tf.get_variable('w4_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w5_x = tf.get_variable('w5_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w6_x = tf.get_variable('w6_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w7_x = tf.get_variable('w7_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w8_x = tf.get_variable('w8_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w9_x = tf.get_variable('w9_x', [state_size], initializer=tf.random_uniform_initializer(0.0,1.0))
    w10_x = tf.get_variable('w10_x', [state_size], initializer=tf.random_uniform_initializer(0.1,1.0))
    w11_x = tf.get_variable('w11_x', [state_size], initializer=tf.random_uniform_initializer(0.1, .1))
    W_x = tf.get_variable('W_x', [state_size, state_size], initializer=tf.random_uniform_initializer(0.0,1.0))

def get_integral(lamb_states):
    ret = 0.0
    with tf.variable_scope('rnn_cell', reuse=True):
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

saver = tf.train.Saver()


t_w1 = np.zeros((state_size),dtype=np.float32)
t_w2 = np.zeros((state_size),dtype=np.float32)
t_w3 = np.zeros((state_size),dtype=np.float32)
t_w4 = np.zeros((state_size),dtype=np.float32)
t_w5 = np.zeros((state_size),dtype=np.float32)
t_w6 = np.zeros((state_size),dtype=np.float32)
t_w7 = np.zeros((state_size),dtype=np.float32)
t_w8 = np.zeros((state_size),dtype=np.float32)
t_w9 = np.zeros((state_size),dtype=np.float32)
t_w10 = np.zeros((state_size),dtype=np.float32)
t_w11 = np.zeros((state_size),dtype=np.float32)
t_W = np.zeros((state_size,state_size),dtype=np.float32)
t_w1_x = np.zeros((state_size),dtype=np.float32)
t_w2_x = np.zeros((state_size),dtype=np.float32)
t_w3_x = np.zeros((state_size),dtype=np.float32)
t_w4_x = np.zeros((state_size),dtype=np.float32)
t_w5_x = np.zeros((state_size),dtype=np.float32)
t_w6_x = np.zeros((state_size),dtype=np.float32)
t_w7_x = np.zeros((state_size),dtype=np.float32)
t_w8_x = np.zeros((state_size),dtype=np.float32)
t_w9_x = np.zeros((state_size),dtype=np.float32)
t_w10_x = np.zeros((state_size),dtype=np.float32)
t_w11_x = np.zeros((state_size),dtype=np.float32)
t_W_x = np.zeros((state_size,state_size),dtype=np.float32)

def train_network(num_epochs,state_size=N, verbose=True):
    with tf.Session() as sess:
        import os
        # if not os.path.isdir(folder+'logs_opinion'+str(N)):
        #     os.mkdir(folder+'logs_opinion'+str(N))
        # train_writer = tf.train.SummaryWriter(folder+'logs_opinion'+str(N), sess.graph)
        sess.run(tf.initialize_all_variables())
        training_losses = []
        if os.path.isfile(model_file):
            saver.restore(sess, model_file)
            training_state = np.zeros((state_size))
            t_w1, t_w2, t_w3, t_w4, t_w5, t_w6, t_w7, t_w8, t_w9, t_w10, t_w11, t_W = sess.run(
                [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,W])
            t_w1_x, t_w2_x, t_w3_x, t_w4_x, t_w5_x, t_w6_x, t_w7_x, t_w8_x, t_w9_x, t_w10_x, t_w11_x,t_W_x = sess.run(
                [w1_x,w2_x,w3_x,w4_x,w5_x,w6_x,w7_x,w8_x,w9_x,w10_x,w11_x,W_x])
        
    return training_losses,t_w1, t_w2, t_w3, t_w4, t_w5, t_w6, t_w7, t_w8, t_w9, t_w10, t_w11, t_W,t_w1_x, t_w2_x, t_w3_x, t_w4_x, t_w5_x, t_w6_x, t_w7_x, t_w8_x, t_w9_x, t_w10_x, t_w11_x,t_W_x

training_losses,t_w1, t_w2, t_w3, t_w4, t_w5, t_w6, t_w7, t_w8, t_w9, t_w10, t_w11, t_W,t_w1_x, t_w2_x, t_w3_x, t_w4_x, t_w5_x, t_w6_x, t_w7_x, t_w8_x, t_w9_x, t_w10_x, t_w11_x,t_W_x = train_network(2000)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def rnn_cell_test(u,t,m,state,state_x):

    sig1 = t_w3 * sigmoid(t_w4 * (m - t_w5))
    sig2 = t_w6 * sigmoid(-t_w7 * (m - t_w8))
    h = sigmoid(t_w1 * np.exp(-t_w2 * t) * state + np.matmul([u], t_W)[0,:] * np.matmul([u], adj)[0,:] * (sig1 - sig2))
    lamb = np.exp(t_w9 + t_w10 * t + t_w11 * h)

    sig1_x = t_w3_x * sigmoid(t_w4_x * (m - t_w5_x))
    sig2_x = t_w6_x * sigmoid(-t_w7_x * (m - t_w8_x))
    h_x = sigmoid(t_w1_x * np.exp(-t_w2_x * t) * state_x + np.matmul([u], t_W_x)[0,:] * np.matmul([u], adj)[0,:] * (sig1_x - sig2_x))
    x = np.tanh(t_w9_x + t_w11_x * h_x)

    return h, lamb, h_x, x


def t_get_integral(t_lamb_states):
    ret = 0.0
    for i in range(batch_size,total_size-1):
        ret += np.add.reduce(np.exp(t_w9+t_w10*dt[i+1]+t_w11*t_lamb_states[i-batch_size]) - np.exp(t_w9+t_w11*t_lamb_states[i-batch_size]) / t_w10)
    return ret

def t_log_lambda(t_lambdas):
    log_sum=0.0
    for i in range(batch_size,total_size):
        log_sum += np.log(np.add.reduce(t_lambdas[i-batch_size]*user[i]))
    return log_sum

def t_get_mse(t_opinions):
    val = 0.0
    mlist = [0.0]    
    for i in range(batch_size,total_size):
        val += np.power(np.add.reduce(t_opinions[i-batch_size]*user[i]) - X[i], 2)
        #mlist.append(val/float(total_size-batch_size))
    return val/(total_size - batch_size)

def t_get_pol(t_opinions):
    total = total_size - batch_size
    inacc = 0
    frac = 0.0
    plist = [0.0]
    for i in range(batch_size,total_size):
        if np.add.reduce(t_opinions[i-batch_size]*user[i])*X[i] < 0.0:
            inacc += 1
            frac = frac + 1.0 / float(total)
        #plist.append(frac)
    return float(inacc)/total

def get_sent(u, G, tm):
    for i in G[u]:
        if i[1]>=tm:
            return i[0]
    return -2.5
def get_sent_2(u,G,tm):
    for i in range(len(G[u])):
        if G[u][i][1] > tm:
            return G[u][i-1][0]
        if G[u][i][1] == tm:
            return G[u][i][0]
    return 0.0
def test_network():
    t_state = np.zeros((state_size))
    t_state_x = np.zeros((state_size))
    t_lambdas = []
    t_lamb_states = []
    t_opinions = []
    t_op_states = []
    #for i in range(batch_size):
        #t_state, t_lamb, t_state_x, t_x = rnn_cell_test(user[i],dt[i],dm[i],t_state,t_state_x)
        #t_lambdas.append(t_lamb)
        #t_lamb_states.append(t_state)
        #t_opinions.append(t_x)
        #t_op_states.append(t_state_x)
    for i in range(batch_size,total_size):
        t_state, t_lamb, t_state_x, t_x = rnn_cell_test(user[i],dt[i],dm[i],t_state,t_state_x)
        t_lambdas.append(t_lamb)
        t_lamb_states.append(t_state)
        t_opinions.append(t_x)
        t_op_states.append(t_state_x)
    #mse_loss,mlist = t_get_mse(t_opinions)
    #pol_loss,plist = t_get_pol(t_opinions)
    #lamb_loss = t_get_integral(t_lamb_states) - t_log_lambda(t_lambdas)
    #test_loss = mse_loss + lamb_loss
    hours = range(11)
    t_last = H[batch_size-1][1]
    print t_last, '\n', H[total_size-1][1], '\n', H[total_size-1][1]-t_last
    #for i in range(batch_size,total_size):
        #hours.append((H[i][1]-t_last)/3600.0) 
    mse_list = []
    ind = 0
    for i in hours:
        sum = 0.0
        tm = t_last + i * 3600
        while ind < (total_size-batch_size) and H[batch_size+ind][1] < tm:
            ind += 1
        num = 0
        for u in range(N):
            g = get_sent(u,G,tm)
            g1 = g
            if ind < (total_size-batch_size):
                #t = H[batch_size+ind][1]
	        #m = H[batch_size+ind][2]
	        #del_m = [0.0] * N
                #for j in range(N):
                    #del_m[j] = abs(m-get_sent_2(u,G,t_last))
                #sig1_x = t_w3_x * sigmoid(t_w4_x * (del_m - t_w5_x))
                #sig2_x = t_w6_x * sigmoid(-t_w7_x * (del_m - t_w8_x))
                #h_x = sigmoid(t_w1_x * np.exp(-t_w2_x * (t-t_last)/3600.0) * t_state_x + np.matmul([user[batch_size+ind]], t_W_x)[0,:] * np.matmul([user[batch_size+ind]], adj)[0,:] * (sig1_x - sig2_x))
                #x = np.tanh(t_w9_x + t_w11_x * h_x)
                #g1 = x[u]
                g1 = t_opinions[ind][u]
            if g > -1:
                sum += (g - g1)**2 
                num += 1
        if num > 0:
            mse_list.append(sum / float(num)) 
        else:
            mse_list.append(sum)
    pol_list = []
    ind = 0
    
    for i in hours:
        sum = 0.0
        tm = t_last + i * 3600
        while ind < len(t_opinions) and H[batch_size+ind][1] < tm:
            ind += 1
        num = 0
        for u in range(N):
            g = get_sent(u,G,tm)
            g1 = g
            l = [i[0] for i in G[u]]
            sm = 0
            for j in l:
             sm += j
            meanG = 0.0
            if len(l) > 0:   meanG = sm / len(l)
            if ind < len(t_opinions):
               g1 = t_opinions[ind][u]
            if g > -1 and (g-0.3) * (g1-0.3) < 0:
               sum += 1.0 
            if g > -1: 
               num += 1
        if num > 0:
            pol_list.append(sum / float(num))
        else:
            pol_list.append(sum)
    mse_loss = t_get_mse(t_opinions)
    pol_loss = t_get_pol(t_opinions)
    print "MSE:",mse_loss," Polarity:",pol_loss 
    return hours, mse_list, pol_list

def get_sent2(u, G, tm):
    for i in G[u]:
        if i[1]>=tm:
            return i[0]
    return -2.5

hours_, mse_list_, pol_list_ = test_network()
#print >> sys.stderr, "(MSE, SE, LAMBDA, TOTAL LOSS, POLARITY ERROR)", mse_loss_/float(total_size-batch_size), mse_loss_,lamb_loss_,test_loss_,pol_loss_
#print "(MSE, SE, LAMBDA, TOTAL LOSS, POLARITY ERROR)", mse_loss_/float(total_size-batch_size), mse_loss_,lamb_loss_,test_loss_,pol_loss_
print 'T = ', hours_, '\nMSE = ', mse_list_, '\nPolarity error = ',pol_list_
end_time = time.time()
print >> sys.stderr, 'Duration', end_time - start_time, 'seconds'
#print "MSE through time:", mlist_, '\nPolarity error through time:', plist_, '\nTime:', hours_


