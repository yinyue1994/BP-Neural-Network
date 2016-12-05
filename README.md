# BP-Neural-Network

import numpy as np
import math
address='C:/Users/Emily/Desktop/Data Mining/assignment5/iris-numeric.txt'
fin=open(address)
iris=np.loadtxt(address,delimiter=',')

address2='C:/Users/Emily/Desktop/Data Mining/assignment5/Concrete_Data_RNorm_Class_train.txt'
fin2=open(address2)
trainset=np.loadtxt(address2,delimiter=',')

address3='C:/Users/Emily/Desktop/Data Mining/assignment5/Concrete_Data_RNorm_Class_test.txt'
fin3=open(address3)
testset=np.loadtxt(address3,delimiter=',')

# logistic sigmoid function
def f(z):
	for i in range(len(z)):
		z[i]=1/(1+math.exp(-z[i]))
	return z

def ann(train,test,n_h,eta,epochs):
	n=train.shape[0]
	d=train.shape[1]-1
	y=train[:,-1]
	mat_x=train[:,0:d]
	mat_x=np.append(mat_x,np.ones((n,1)),axis=1)
	n_i=d+1
	factory,indices=np.unique(y,return_inverse=True)
	n_o=len(factory)
	vec_y=np.zeros((n,n_o))
	for i in range(len(y)):
		vec_y[i,indices[i]]=1
	mat_w1=np.random.uniform(-0.1,0.1,(n_i,n_h))
	mat_w2=np.random.uniform(-0.1,0.1,(n_h+1,n_o))
	y_est=np.zeros((n,n_o))
	for e in range(epochs):
		rand=list(range(n))
		np.random.shuffle(rand)
		for i in rand:
			net_h=np.dot(mat_w1.T,mat_x[i])
			out_h=f(net_h)
			out_h=np.append(out_h,[1])
			net_o=np.dot(mat_w2.T,out_h.T)
			out_o=f(net_o)
			y_est[i]=out_o
			error=0.5*np.dot((y_est[i]-vec_y[i]),(y_est[i]-vec_y[i]).T)
			if error>0:
				delta_j2=y_est[i].reshape((n_o,1))*(np.ones((n_o,1))-y_est[i].reshape((n_o,1)))*(vec_y[i].reshape((n_o,1))-y_est[i].reshape((n_o,1)))
				delta_w2=eta*out_h.reshape((n_h+1,1))*delta_j2.reshape((1,n_o))
				mat_w2=mat_w2+delta_w2
				sigma=np.dot(mat_w2,delta_j2.reshape((n_o,1)))
				delta_j1=out_h.reshape((n_h+1,1))*(np.ones((n_h+1,1))-out_h.reshape((n_h+1,1)))*sigma
				delta_w1=eta*mat_x[i].reshape((n_i,1))*delta_j1[:-1,:].reshape((1,n_h))
				mat_w1=mat_w1+delta_w1
	print('mat_w1','\n',mat_w1)
	print('mat_w2','\n',mat_w2)
	n_t=test.shape[0]
	y_t=test[:,-1]
	mat_xt=test[:,0:d]
	mat_xt=np.append(mat_xt,np.ones((n_t,1)),axis=1)
	factory_t,indices_t=np.unique(y_t,return_inverse=True)
	n_ot=len(factory_t)
	vec_yt=np.zeros((n_t,n_ot))
	y_est_t=np.zeros((n_t,n_ot))
	errornum=0
	for i in range(len(y_t)):
		vec_yt[i,indices_t[i]]=1
		net_ht=np.dot(mat_w1.T,mat_xt[i])
		out_ht=f(net_ht)
		out_ht=np.append(out_ht,[1])
		net_ot=np.dot(mat_w2.T,out_ht.T)
		out_ot=f(net_ot)
		y_est_t[i]=out_ot
		maximum=np.max(y_est_t[i])
		for j in range(len(y_est_t[i])):
			if y_est_t[i,j]==maximum: y_est_t[i,j]=1
			else: y_est_t[i,j]=0
		dif=np.array_equal(vec_yt[i],y_est_t[i])
		if dif==False: errornum += 1
	acc=1-errornum/n_t
	print('accuracy','\n',acc)

ann(trainset,testset,5,0.1,500)
