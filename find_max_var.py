import numpy as np
import tensorflow as tf
import utility2
import sys
import time

start_time = time.time()
print >> sys.stderr, 'Start time', start_time
N = int(sys.argv[1])
folder = ''
if len(sys.argv)>2:
        folder = sys.argv[2]
        if not folder[-1]=='/':
                folder = folder + '/'
#model_file = sys.argv[3]
THRESHOLD = 0.1
hidden_layer_size = N
input_size = N
target_size = 1

inv_map = utility2.get_id_map(folder+'good_id_to_old_id.json',N)
adj = utility2.get_adj(folder+'edgelist.txt', inv_map, N)
H = utility2.get_history(folder+'opinion.txt', inv_map, N)
G = utility2.make_adj_H(H,N)

var = []
for u in range(N):
  var.append((u, np.var([i[0] for i in G[u]]))) 

var = sorted(var, key= lambda x: x[1])
print var[:10]
