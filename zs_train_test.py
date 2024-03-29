import numpy as np
import os 
import sys
import argparse
import pdb
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

debug=False

import sys
sys.path.append('./models')
#from models import lenet
from models import vgg
#from models import resnet
from config import cfg

torch.manual_seed(0)

def init_models(arch, precision, retrain, checkpoint_path):

    in_channels = 3

    """ unperturbed model 
    """
    if arch == 'vgg11':
      model  = vgg('A',in_channels, 10, True, precision)
    elif arch == 'vgg16':
      model  = vgg('D',in_channels, 10, True, precision)
    elif arch == 'resnet18':
      model = resnet('resnet18', 10, precision) 
    elif arch == 'resnet34':
      model = resnet('resnet34', 10, precision) 
    else:
      model = lenet(in_channels,10,precision)
    
    print(model)
    # Fixed initial condition
    if cfg.epsi > 1.e-8:
        with torch.no_grad():
            for i, lay_i in enumerate(model.features):
                if type(lay_i) == nn.Conv2d:
                    file_w = 'weights/wt_' + str(i) + '.npy'
                    file_b = 'weights/b_' + str(i) + '.npy'
                    with open(file_w, 'rb') as fw:
                        temp_arr = torch.from_numpy(np.load(fw))
                        dims_w = temp_arr.shape
                        for d1 in range(dims_w[0]):
                            for d2 in range(dims_w[1]):
                                for d3 in range(dims_w[2]):
                                    for d4 in range(dims_w[3]):
                                        model.features[i].weight[d1,d2,d3,d4] = temp_arr[d1,d2,d3,d4]
                                        if i == 40 and d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
                                            print(lay_i.weight[d1,d2,d3,d4])
                                            #model.features[i].weight[d1,d2,d3,d4] += cfg.epsi*np.random.rand()
                                        
                    with open(file_b, 'rb') as fb:
                        temp_arr = torch.from_numpy(np.load(fb))
                        dims_b = temp_arr.shape
                        for d1 in range(dims_b[0]):
                            model.features[i].bias[d1] = temp_arr[d1]

    checkpoint_epoch = 0
    if (retrain):
        print('Restoring model from checkpoint', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        print('restored checkpoint at epoch - ',checkpoint['epoch'])
        print('Training loss =', checkpoint['loss'])
        print('Training accuracy =', checkpoint['accuracy'])
        checkpoint_epoch=checkpoint['epoch']

    return model, checkpoint_epoch

def train_test(trainloader, testloader, arch, dataset, precision, retrain, checkpoint_path, device):

    loss_list = np.zeros(cfg.epochs)
    #norm_list = np.zeros(cfg.epochs)
    #norm1_list =  np.zeros(cfg.epochs)
    acc_list =  np.zeros(cfg.epochs)
    ntbs = len(testloader)
    test_loss_list = np.zeros(ntbs*cfg.nt)
    y = 0
    model, checkpoint_epoch = init_models(arch, precision, retrain, checkpoint_path)
    print('Training with Learning rate %.4f'%(cfg.learning_rate))
    #print('Model weight is ....', model.features[0].weight[0,0,0,0])
    opt = optim.SGD(model.parameters(),lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)

    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    pflag = 0
    #curr_lr=cfg.learning_rate
                
    if cfg.noise > 1.e-8:
        outputs_corrupt = np.zeros((len(trainloader), cfg.batch_size))
    for x in range(cfg.epochs):

        running_loss = 0.0
        running_correct = 0
        pflag = 0
        for batch_id, (inputs,outputs) in enumerate(trainloader):
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            opt.zero_grad()
            
            # Sample noisy labels
            if x == 0 and cfg.noise > 1.e-8:
                for ind, o_ind in enumerate(outputs): 
                    u = np.random.rand()
                    outputs_corrupt[batch_id, ind] = o_ind
                    if u < cfg.noise:
                        u = np.random.rand()
                        bin_u = int(9*u)
                        if bin_u < o_ind:
                            outputs_corrupt[batch_id, ind] = bin_u
                        else:
                            outputs_corrupt[batch_id, ind] = bin_u+1

            # Inject noise into labels
            if cfg.noise > 1.e-8:
                for ind, o_ind in enumerate(outputs):
                    outputs[ind] = outputs_corrupt[batch_id,ind]
            
            # Input perturbation
            if batch_id == 0 and cfg.pert_ip == 1:
                first_ip = inputs[0,:,:,:]
                first_op = outputs[0]
                inputs = inputs[1:,:,:,:]
                outputs = outputs[1:]
                inputs[2,:,:,:] = first_ip
                outputs[2] = first_op

                #    inputs[0,:,:,:] += inputs.mean()/1000*(-1.0 + 2.0*np.random.rand())
            
            model_outputs = model(inputs)
            _, preds = torch.max(model_outputs, 1)
            outputs = outputs.view(outputs.size(0))  ## changing the size from (batch_size,1) to batch_size. 

            loss = nn.CrossEntropyLoss()(model_outputs, outputs)

            if (debug):
                print('epoch %d batch %d loss %.3f'%(x,batch_id,loss))

            # Compute gradient of perturbed weights with perturbed loss 
            loss.backward()

            # update restored weights with gradient 
            opt.step()


            running_loss+=loss.item()
            running_correct+=torch.sum(preds == outputs.data)
        # update learning rate
        #if ((x==80) or (x == 120)):
        #    curr_lr /= 10.0
        #    for param_group in opt.param_groups:
        #        param_group['lr'] = curr_lr
        #    print('Training with Learning rate %.4f'%(curr_lr))

        loss_list[x] = (running_loss/(batch_id))

        # norm
        #norm_list[x] = (model.features[0].weight.view(-1).square().sum().detach().cpu().numpy())
        #norm1_list[x] = (model.features[0].weight[0,0,0,0].detach().cpu().numpy())

        accuracy = running_correct.double()/(len(trainloader.dataset))
        print('epoch %d loss %.6f accuracy %.6f' %(x, running_loss/(batch_id), accuracy))
        #writer.add_scalar('Loss/train', running_loss/batch_id, x)   ## loss/#batches 
        
        acc_list[x], test_loss = test(testloader, model, device)
        if x > cfg.runup:
            test_loss_list += test_loss/(cfg.epochs-cfg.runup)
        if x == cfg.epochs-1:
            model_path = arch + '_' + dataset + '_' + str(checkpoint_epoch+x) + 'p3n25' + '.pth'
            torch.save({'epoch': (checkpoint_epoch+x), 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'loss': running_loss/batch_id, 'accuracy': accuracy}, model_path)
                #utils.collect_gradients(params, faulty_layers)
    name_of_folder = "outputs/stoc_stab_final/pert3_noise_25/"
    
    with open(name_of_folder + "test_loss.txt", 'a') as f:
        np.savetxt(f, acc_list)
    with open(name_of_folder + "loss.txt", 'a') as f:
        np.savetxt(f, loss_list)
    with open(name_of_folder + "test_loss_cumsum.txt", 'a') as f:
        np.savetxt(f, test_loss_list)



           
def test(testloader, model, device):            
    model.eval()
    #running_correct = 0.0
    test_loss_mean = 0.0
    ntbs = len(testloader)
    test_loss_list = np.zeros(cfg.nt*ntbs)
    loss = nn.CrossEntropyLoss(reduction='none')
    mean_loss = nn.CrossEntropyLoss()

    with torch.no_grad():

      for t, (inputs,classes) in enumerate(testloader):

          inputs = inputs.to(device)
          classes = classes.to(device)
          model_outputs =model(inputs)
          
          if t < cfg.nt:
              test_loss = loss(model_outputs, classes)
              test_loss_list[ntbs*t:ntbs*(t+1)] = test_loss.cpu().detach().numpy()
          test_loss_mean += cfg.test_batch_size*mean_loss(model_outputs, classes)

          #lg, preds = torch.max(model_outputs, 1)
          #correct=torch.sum(preds == classes.data)
          #running_correct += correct
          
    ltest = len(testloader.dataset)
    #acc = (running_correct.double()/ltest)
    test_loss_mean /= ltest
    #print('Eval Accuracy %.3f'%(acc))
    return test_loss_mean, test_loss_list



