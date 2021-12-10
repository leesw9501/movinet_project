import cv2
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
#import transforms as T
from movinets.config import _C
import numpy as np
from movinets import MoViNet
import random
import gc

torch.manual_seed(97)
num_frames = 16 # 16
clip_steps = 2
Bs_Train = 32
Bs_Test = 32

class_dict=dict()
class_dict['normal']=0
class_dict['assault']=1
class_dict['fight']=2
class_dict['burglary']=3
class_dict['vandalism']=4
class_dict['swoon']=5
class_dict['wander']=6
class_dict['trespass']=7
class_dict['dump']=8
class_dict['robbery']=9
class_dict['datefight']=10
class_dict['kidnap']=11
class_dict['drunken']=12

root = '../pre_pro_data/'

def make_data():
    train_data = []
    #test_data = np.array(test_data)

    train_label = np.empty((0), 'float')

    for class_num in range(1,13):
        #print(class_num, len(train_data))
        #print(gc.collect())
        for num in range(120):
            path = root + str(class_num) + '/' + str(num)
            cap = cv2.VideoCapture(path+'.mp4')
            if cap.isOpened():
                train_label = np.concatenate((train_label, np.load(path + '.npy')[num_frames-1:]))
                img_arr=[]
                while True:
                    ret, img = cap.read()
                    if ret:
                        img_arr.append(img)
                        if len(img_arr)==num_frames:
                            train_data.append(img_arr.copy())
                            del img_arr[0]
                    else:
                        break

                cap.release()

            else:
                print('cannot open the file', path+'.mp4')
                break
                
    return train_data, train_label

x_train, y_train = make_data()

def make_test():
    train_data = []
    #test_data = np.array(test_data)

    train_label = np.empty((0), 'float')

    for class_num in range(1,13):
        #print(class_num, len(train_data))
        #print(gc.collect())
        for num in range(120, 124):
            path = root + str(class_num) + '/' + str(num)
            cap = cv2.VideoCapture(path+'.mp4')
            if cap.isOpened():
                train_label = np.concatenate((train_label, np.load(path + '.npy')[num_frames-1:]))
                img_arr=[]
                while True:
                    ret, img = cap.read()
                    if ret:
                        img_arr.append(img)
                        if len(img_arr)==num_frames:
                            train_data.append(img_arr.copy())
                            del img_arr[0]
                    else:
                        break

                cap.release()

            else:
                print('cannot open the file', path+'.mp4')
                break
                
    return train_data, train_label

x_valid, y_valid = make_test()

a=dict()
for i in range(13):
    a[str(i)]=0
for i in y_train:
    a[str(int(i))]+=1

nSamples = []
for s in a:
    nSamples.append(a[s])

normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
normedWeights = torch.FloatTensor(normedWeights).cuda()

class cctv():

    def __init__(self, train_data, label_data):
        self.train = train_data
        self.label = torch.from_numpy(label_data)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.train[idx]).reshape(3, 16, 172, 172)).float()/255, self.label[idx].long()

cctv_train = cctv(x_train, y_train)
train_loader = DataLoader(cctv_train, batch_size=Bs_Train, shuffle=True)

cctv_val = cctv(x_valid, y_valid)
val_loader = DataLoader(cctv_val, batch_size=Bs_Train, shuffle=False)

def train_iter_stream(model, optimz, data_load, loss_val, n_clips = 2, n_clip_frames=8):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames. 
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.
    
    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    #clean the buffer of activations
    samples = len(data_load.dataset)
    model.cuda()
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()
    samples2 = 0
    csamp = 0
    
    for i, (data,target) in enumerate(data_load):
        data = data.cuda()
        target = target.cuda()
        l_batch = 0
        #backward pass for each clip
        for j in range(n_clips):
            #output = F.log_softmax(model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
            #loss = F.nll_loss(output, target)
            #print('asdasd', model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]).shape)
            #print('output', output.shape)
            #print('target', target.shape)
            
            output = model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)])
            loss = torch.nn.CrossEntropyLoss( weight = normedWeights )
            loss = loss(output, target)/n_clips
            
            _, pred = torch.max(output, dim=1)
            #loss = F.nll_loss(output, target)/n_clips
            loss.backward()
        csamp += pred.eq(target).sum()
        samples2 += len(target)
        l_batch += loss.item()*n_clips
        optimz.step()
        optimz.zero_grad()
        
        #clean the buffer of activations
        model.clean_activation_buffers()
        if i % 200 == 0:
            gc.collect()
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch)+
                 '  Accuracy:' + '{:5}'.format(csamp) + '/' +
                  '{:5}'.format(samples2) + ' (' +
                  '{:4.2f}'.format(100.0 * csamp / samples2) + '%)')
            loss_val.append(l_batch)

def evaluate_stream(model, data_load, loss_val, n_clips = 2, n_clip_frames=8):
    model.eval()
    model.cuda()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data, target in data_load:
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)])
                loss = torch.nn.CrossEntropyLoss( weight = normedWeights )
                loss = loss(output, target)/n_clips
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()

    aloss = tloss /  len(data_load)
    loss_val.append(aloss)
    print('Average loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)')
    return '{:4.2f}'.format(100.0 * csamp / samples)

N_EPOCHS = 3
start_time = time.time()
model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )


trloss_val, tsloss_val = [], []
model.classifier[3] = torch.nn.Conv3d(2048, 13, (1,1,1))
optimz = optim.Adam(model.parameters(), lr=0.00005)


for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_time = time.time()
    train_iter_stream(model, optimz, train_loader, trloss_val)
    print('Train time:', '{:5.2f}'.format(time.time() - train_time), 'seconds')
    
    print('\nValidation result')
    test_time = time.time()
    stracc = evaluate_stream(model, val_loader, tsloss_val)
    print('Validation time:', '{:5.2f}'.format(time.time() - test_time), 'seconds\n')
    
    torch.save(model.state_dict(), './model/epoch_'+str(epoch)+'_acc_'+stracc+'.pth')
    

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')