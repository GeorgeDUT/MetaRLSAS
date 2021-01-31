"""
need env: rl-basic
"""
import numpy as np
result=np.load('results.npz',allow_pickle=True)
print('files',result.files)
# task=result['tasks']
train_return=result['train_returns']
valid_return=result['valid_returns']
# print(task)
print(train_return)
print(valid_return)


# 任务长度
'''
print('task-lenth',len(task))
print('task',task)
'''

'''
print('task',train_return[0])
print('task',valid_return[0])
'''

"""
t_r=[0,0,0,0,0,0,0,0,0,0]
v_r=[0,0,0,0,0,0,0,0,0,0]
for i in range(len(train_return)):
    t_r=np.sum([t_r,train_return[i]],axis = 0)
for i in range(len(valid_return)):
    v_r=np.sum([v_r,valid_return[i]],axis = 0)
for i in range(10):
    t_r[i]=t_r[i]/(1.0*len(train_return))
    v_r[i]=v_r[i]/(1.0*len(train_return))
print('train_r',t_r,'lenth',len(train_return))
print('valid_r',v_r,'lenth',len(valid_return))
"""

t_r=np.zeros(len(train_return),dtype=float)
v_r=np.zeros(len(valid_return),dtype=float)

for i in range(len(train_return)):
    t_r[i]=np.sum(train_return[i])

for i in range(len(valid_return)):
    v_r[i]=np.sum(valid_return[i])

allsum=0
for i in range(len(train_return)):
    allsum=allsum+len(train_return[i])


print('avg_train_r',sum(t_r)/(1.0*allsum),'lenth',len(train_return),'all_episode',allsum)


print('avg_valid_r',sum(v_r)/(1.0*allsum),'lenth',len(valid_return),'all_episode',allsum)

print('avg_dis_r',sum((v_r-t_r)/(1.0*allsum)))

print('all_train_r',t_r)
print('all_valid_r',v_r)
print('dis_r',(v_r-t_r)/(1.0*len(train_return[i])),'lenth',len(train_return))


