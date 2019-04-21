import math
import numpy as np
import random
from matplotlib import pyplot as plt
import cv2

def sigmoid(w,b,x):
    z=w*x+b
    sig= 1/( 1+np.exp(-z))
    return sig
def cost(w,b,x_list,gt_y_list):
    avg_loss=0
    m=len(gt_y_list)
    for i in range(len(x_list)):
        a=sigmoid(w,b,x_list[i])
        avg_loss+=gt_y_list[i]*np.log(a)+(1-gt_y_list[i])*np.log(1-a)
    avg_loss=-(avg_loss/m)
    return avg_loss
def gradientDescent(sig,gty,x):
    diff=sig-gty
    dw=diff*x
    db=diff
    return dw,db
def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    avg_dw,avg_db=0,0
    batch_size = len(batch_x_list)
    # print(bat)
    for i in range(batch_size):
        sig = sigmoid(w, b, batch_x_list[i])  # get label data
        dw, db = gradientDescent(sig, batch_gt_y_list[i], batch_x_list[i])
        avg_dw += dw
        avg_db += db
    avg_dw /= batch_size
    avg_db /= batch_size
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b
def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(cost(w, b, x_list, gt_y_list)))
        min_loss = cost(w, b, x_list, gt_y_list)
        while min_loss<0.1:
            return
def gen_sample_data():
    w =  random.randint(-40, 40)+random.random()	# for noise random.random[0, 1)
    b =  random.randint(-20, 20) +random.random()
    num_samples = 100
    x_list = []
    y_list = []
    mid=num_samples//2
    area = np.random.rand(5) * 100
    for i in range(num_samples):
        if i <= mid:
            y = ((random.randint(50, 99)) + random.random()) / 100
            x = random.randint(-45,0) + random.randint(-5, 5) * random.random()
            x_list.append(x)
            y_list.append(y)
            plt.scatter(x, y, s=area, c='red', alpha=1)
        else:
            y = ((random.randint(0,49)) + random.random()) / 50
            x = random.randint(0,45) + random.randint(-5, 5) * random.random()
            x_list.append(x)
            y_list.append(y)
            plt.scatter(x, y, s=area, c='green', alpha=1)

        plt.show
        return x_list, y_list, w, b

def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 50, lr, max_iter)





if __name__=='__main__':
    run()