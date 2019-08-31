# coding: utf-8
from __future__ import division
import os
import random
import sys
import argparse
import shutil
import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import dataset
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_live_rgb_dir   = './data/live_train_face_rgb'
train_live_depth_dir = './data/live_train_face_depth'
train_fake_rgb_dir   = './data/fake_train_face_rgb'

test_live_rgb_dir    = './data/live_test_face_rgb'
test_fake_rgb_dir    = './data/fake_test_face_rgb'

parser = argparse.ArgumentParser(description='Tensoeflow Liveness Training')
parser.add_argument('--scale', default=1.0, type=float,
                    metavar='N', help='net scale')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='./model', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args([])

class Net():
    def __init__(self, scale = 1.0,expand_ratio=1):        
        self.scale = scale
        
    def res(self,x):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv_bn(x,inp, oup):
            W = weight_variable((3,3,inp,oup))
            t = tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
            t = tf.layers.batch_normalization(t)
            t = tf.nn.leaky_relu(t,alpha=0.25)
            return t

        def conv_dw(x,inp, oup,stride = 1):
            W1 = weight_variable((3,3,inp,inp))
            t = tf.nn.conv2d(x,W1,strides=[1,stride,stride,1],padding='SAME')
            t = tf.layers.batch_normalization(t)
            t = tf.nn.leaky_relu(t,alpha=0.25)
            W2 = weight_variable((1,1,inp,oup))
            t = tf.nn.conv2d(t,W2,strides=[1,1,1,1],padding='SAME')
            t = tf.layers.batch_normalization(t)
            t = tf.nn.leaky_relu(t,alpha=0.25)
            return t

        def step1(x):
            with tf.variable_scope('step1'):
                x = conv_dw(x,(int)(32 * self.scale), (int)(64 * self.scale), 2)
                x = conv_dw(x,(int)(64 * self.scale), (int)(128 * self.scale))
                x = conv_dw(x,(int)(128 * self.scale), (int)(128 * self.scale))
            return x

        def step1_shotcut(x):
            with tf.variable_scope('step1_shotcut'):
                return conv_dw(x,(int)(32 * self.scale), (int)(128 * self.scale), 2)

        def step2(x):
            with tf.variable_scope('step2'):
                x = conv_dw(x,(int)(128 * self.scale), (int)(128 * self.scale), 2)
                x = conv_dw(x,(int)(128 * self.scale), (int)(256 * self.scale))
                x = conv_dw(x,(int)(256 * self.scale), (int)(256 * self.scale))
            return x

        def step2_shotcut(x):
            with tf.variable_scope('step2_shotcut'):
                return conv_dw(x,(int)(128 * self.scale), (int)(256 * self.scale), 2)

        def depth_ret(x):
            with tf.variable_scope('depth_ret'):
                W_d1 = weight_variable((3,3,256,256))
                x = tf.nn.conv2d(x,W_d1,strides=[1,1,1,1],padding='SAME')
                x = tf.layers.batch_normalization(x)
                W_d2 = weight_variable((1,1,256,2))
                x = tf.nn.conv2d(x,W_d2,strides=[1,1,1,1],padding='SAME')
            return x
        def depth_shotcut(x):
            with tf.variable_scope('depth_shotcut'):
                return conv_dw(x,(int)(256 * self.scale), 2)

        self.softmax = tf.nn.softmax
        self.dropout = tf.nn.dropout

        def forward(x):
            head = conv_bn(x,3,32)
            step1_ = step1(head) + step1_shotcut(head)
            step2_ = self.dropout(step2(step1_) + step2_shotcut(step1_),keep_prob=0.5)
            depth = self.softmax(depth_ret(step2_))
            class_pre = depth_shotcut(step2_) + depth
            class_pre = tf.reshape(class_pre,(-1, 2048))
            class_ret = tf.layers.dense(class_pre,2)
            return depth, class_ret
        return forward(x)

class Model(object):
    def __init__(self,net,sess,criterion_depth, criterion_class, optimizer):
        self.net = net
        self.sess = sess
        self.criterion_depth = criterion_depth
        self.criterion_class = criterion_class
        self.optimizer = optimizer              
        
        self.inputs = tf.placeholder(tf.float32,[None,128,128,3])
        self.depths = tf.placeholder(tf.float32,[None,1,32,32])
        self.labels = tf.placeholder(tf.int32,[None])
        self.one_hot_label = tf.one_hot(self.labels,2)
        output, class_ret = net.res(self.inputs)
        
        '''
        train
        '''
        out_depth = tf.reshape(output[:,:,:,0],(-1,1,32,32))
        self.loss_depth = self.conv_loss(out_depth, self.depths, self.criterion_depth)
        self.loss_depth_mean = tf.reduce_mean(self.loss_depth)
        self.loss_class = self.criterion_class.forward(class_ret, self.one_hot_label)
        self.loss_class_mean = tf.reduce_mean(self.loss_class)
        loss = self.loss_depth + self.loss_class
        self.train_step = optimizer.minimize(loss)
           
        '''
        validate
        '''

        self.out_depth_val = tf.reshape(output[:,:,:,0],(32,32))
        self.class_output_val = tf.nn.softmax(class_ret)
        self.sess.run(tf.global_variables_initializer())
    def conv_loss(self,out_depth, label_depth, criterion_depth):
        loss0 = criterion_depth.forward(out_depth, label_depth)
        filters1 = tf.Variable([[[[-1, 0, 0],[0, 1, 0],[0, 0, 0]]]], dtype=tf.float32)
        filters2 = tf.Variable([[[[0, -1, 0],[0, 1, 0],[0, 0, 0]]]], dtype=tf.float32)
        filters3 = tf.Variable([[[[0, 0, -1],[0, 1, 0],[0, 0, 0]]]], dtype=tf.float32)
        filters4 = tf.Variable([[[[0, 0, 0],[-1, 1, 0],[0, 0, 0]]]], dtype=tf.float32)
        filters5 = tf.Variable([[[[0, 0, 0],[0, 1, -1],[0, 0, 0]]]], dtype=tf.float32)
        filters6 = tf.Variable([[[[0, 0, 0],[0, 1, 0],[-1, 0, 0]]]], dtype=tf.float32)
        filters7 = tf.Variable([[[[0, 0, 0],[0, 1, 0],[0, -1, 0]]]], dtype=tf.float32)
        filters8 = tf.Variable([[[[0, 0, 0],[0, 1, 0],[0, 0, -1]]]], dtype=tf.float32)

        filters1 = tf.reshape(filters1,(3,3,1,1))
        filters2 = tf.reshape(filters2,(3,3,1,1))
        filters3 = tf.reshape(filters3,(3,3,1,1))
        filters4 = tf.reshape(filters4,(3,3,1,1))
        filters5 = tf.reshape(filters5,(3,3,1,1))
        filters6 = tf.reshape(filters6,(3,3,1,1))
        filters7 = tf.reshape(filters7,(3,3,1,1))
        filters8 = tf.reshape(filters8,(3,3,1,1))

        loss1 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters1, strides = [1,1,1,1],data_format="NCHW",padding ='SAME'),
            tf.nn.conv2d(label_depth, filters1, strides = [1,1,1,1],data_format="NCHW",padding = 'SAME'))
        loss2 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters2,strides = [1,1,1,1],data_format="NCHW", padding ='SAME'),
            tf.nn.conv2d(label_depth, filters2,strides = [1,1,1,1],data_format="NCHW", padding = 'SAME'))
        loss3 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters3, strides = [1,1,1,1],data_format="NCHW",padding ='SAME'),
            tf.nn.conv2d(label_depth, filters1,strides = [1,1,1,1], data_format="NCHW",padding = 'SAME'))
        loss4 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters4,strides = [1,1,1,1],data_format="NCHW", padding ='SAME'),
            tf.nn.conv2d(label_depth, filters4,strides = [1,1,1,1],data_format="NCHW", padding = 'SAME'))
        loss5 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters5,strides = [1,1,1,1],data_format="NCHW", padding ='SAME'),
            tf.nn.conv2d(label_depth, filters5,strides = [1,1,1,1],data_format="NCHW", padding = 'SAME'))
        loss6 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters6,strides = [1,1,1,1], data_format="NCHW",padding ='SAME'),
            tf.nn.conv2d(label_depth, filters6,strides = [1,1,1,1],data_format="NCHW", padding = 'SAME'))
        loss7 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters7,strides = [1,1,1,1],data_format="NCHW", padding ='SAME'),
            tf.nn.conv2d(label_depth, filters7,strides = [1,1,1,1],data_format="NCHW", padding = 'SAME'))
        loss8 = criterion_depth.forward(tf.nn.conv2d(out_depth, filters8,strides = [1,1,1,1], data_format="NCHW",padding ='SAME'),
            tf.nn.conv2d(label_depth, filters8,strides = [1,1,1,1],data_format="NCHW", padding = 'SAME'))

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
        return loss
           
    def train(self, train_loader, epoch):
        losses_depth = AverageMeter()
        losses_class = AverageMeter()
        print("Training...\n")
        for i, data in enumerate(train_loader):
            input, depth, label = data
            input = input.reshape(-1,128,128,3).data.numpy()
            depth = depth.data.numpy()
            label = label.data.numpy()
            loss_depth_,loss_depth_mean_,loss_class_,loss_class_mean_,_ = self.sess.run([self.loss_depth,
                                                                                self.loss_depth_mean,
                                                                                self.loss_class,
                                                                                self.loss_class_mean,self.train_step],
                                                                                feed_dict={self.inputs:input,
                                                                                          self.depths:depth,
                                                                                          self.labels:label})
            if i % 10 == 0:
                print("epoch:{} batch:{} depth loss:{:f} depth avg loss:{:f} class loss:{:f} class avg loss:{:f}".format(
                    epoch, i, loss_depth_, loss_depth_mean_, loss_class_, loss_class_mean_))
                
    def validate(self,val_loader, depth_dir = './depth_predict'):
        try:
            shutil.rmtree(depth_dir)
        except:
            pass
        try:
            os.makedirs(depth_dir)
        except:
            pass
        toImage = standard_transforms.ToPILImage()
        live_scores = []
        fake_scores = []
        
        for i, data in enumerate(val_loader):
            input, label = data
            input = input.reshape(-1,128,128,3).data.numpy()
            label = label.data.numpy()
            out_depth_,class_output_= self.sess.run([self.out_depth_val,self.class_output_val],feed_dict={self.inputs:input})
            image = toImage(out_depth_)
            image = image.convert("L")
            score = class_output_[0][1]
            if label == 0:
                fake_scores.append(score)
                name = '' + depth_dir + '/fake-' + str(i) + '.bmp'
                image.save(name)
            if label == 1:
                live_scores.append(score)
                name = '' + depth_dir + '/live-' + str(i) + '.bmp'
                image.save(name)
        live_scores.sort()
        fake_scores.sort(reverse=True)
        fake_error = 0
        live_error = 0
        for val in fake_scores:
            if val >= 0.50:
                fake_error += 1
            else:
                break
        for val in live_scores:
            if val <= 0.50:
                live_error += 1
            else:
                break
        print('len(fake_scores): ',len(fake_scores))
        print('fake_error: ',fake_error)
        print('len(live_scores): ',len(live_scores))
        print('live_error: ',live_error)
        print('threshold 0.5: lanjielv = ', (len(fake_scores)-fake_error) / len(fake_scores),  '; tongguolv = ', (len(live_scores) - live_error) / len(live_scores))
        print('threshold 0.5: fpr= ', fake_error / len(fake_scores),  '; tpr = ', (len(live_scores) - live_error) / len(live_scores))

def save_checkpoint(saver,sess,path='checkpoint',global_step=None):  
    saver.save(sess,path,global_step)
    
def load_checkpoint(saver,sess,path='checkpoint'):
    saver.restore(sess,tf.train.latest_checkpoint(path))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FocalLoss():
    def __init__(self, gamma = 2, eps = 1e-7):
        self.gamma = gamma
        self.eps = eps
        self.ce = tf.losses.softmax_cross_entropy

    def forward(self, input, target):
        logp = self.ce(target,input)
        p = tf.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return tf.reduce_mean(loss)

class DepthFocalLoss():
    def __init__(self, gamma = 1, eps = 1e-7):       
        self.gamma = gamma
        self.eps = eps
        self.ce = tf.losses.mean_squared_error

    def forward(self, input, target):
        loss = self.ce(target,input)
        loss = (loss) ** self.gamma
        return tf.reduce_mean(loss)

def main(args):
    print('main')
    net = Net(args.scale)  
    print("start load train data")
    normalize = standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    random_input_transform = standard_transforms.Compose([
        standard_transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        standard_transforms.RandomResizedCrop((128, 128), scale=(0.9, 1), ratio=(1, 1)),
        standard_transforms.ToTensor(),
        normalize
    ])

    target_transform = standard_transforms.Compose([
        standard_transforms.Resize((32, 32)),
        standard_transforms.ToTensor()
    ])

    train_set = dataset.Dataset('train', train_live_rgb_dir, train_live_depth_dir, train_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = target_transform)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 4, shuffle = True, drop_last=True)

    val_set = dataset.Dataset('test', test_live_rgb_dir, None, test_fake_rgb_dir,
        random_transform = random_input_transform, target_transform = target_transform)
    val_loader = DataLoader(val_set, batch_size = 1, num_workers = 4, shuffle = False)

    criterion_class = FocalLoss()
    criterion_depth = DepthFocalLoss()
    optimizer = tf.train.AdamOptimizer()   
    sess = tf.Session(config=config)
    model = Model(net,sess,criterion_depth,criterion_class,optimizer)

    summary = tf.summary.FileWriter('./log',tf.get_default_graph())
    saver = tf.train.Saver()
    if args.resume:
        if os.path.isdir(args.resume):     
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = tf.train.get_checkpoint_state(args.resume)
            load_checkpoint(saver,model.sess,args.resume)       
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        model.validate(val_loader, args.arch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        model.train(train_loader,epoch)
        model.validate(val_loader)    
        if (epoch+1) % 1 == 0:        
            save_checkpoint(saver,model.sess,'./model/tf_model',global_step=epoch)
    summary.close()
    model.sess.close()

if __name__ == '__main__':
    main(args)

