import numpy as np
result=np.load('results.npz',allow_pickle=True)
print('files',result.files)
# task=result['tasks']
train_return=result['train_returns']
valid_return=result['valid_returns']
# print(task)
# print(result['num_iterations'])
# print(train_return)
# print(valid_return)
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
