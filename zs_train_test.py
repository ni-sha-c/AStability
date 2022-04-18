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
    norm_list = np.zeros(cfg.epochs)
    norm1_list =  np.zeros(cfg.epochs)
    
    num_acc = 40
    acc_list =  np.zeros(cfg.epochs//num_acc + 2)
    y = 0
    model, checkpoint_epoch = init_models(arch, precision, retrain, checkpoint_path)

    print('Training with Learning rate %.4f'%(cfg.learning_rate))
    opt = optim.SGD(model.parameters(),lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)

    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    pflag = 0
    #curr_lr=cfg.learning_rate
                
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
            if x == 0:
                for ind, o_ind in enumerate(outputs): 
                    u = np.random.rand()
                    outputs_corrupt[batch_id, ind] = o_ind
                    if u < cfg.noise:
                        u = np.random.rand()
                        bin_u = int(9*u)
                        if bin_u < o_ind:
                            outputs_corrupt[batch_id, ind] = bin_u
                        else:
                            outputs_corrupt[batch_id, ind] = bin_u

            # Inject noise into labels
            for ind, o_ind in enumerate(outputs):
                outputs[ind] = outputs_corrupt[batch_id,ind]
            
            # Input perturbation
            if batch_id == 0:
                #first_ip = inputs[0,:,:,:]
                #first_op = outputs[0]
                inputs = inputs[1:,:,:,:]
                outputs = outputs[1:]
                #inputs[4,:,:,:] = first_ip
                #outputs[4] = first_op

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
        norm_list[x] = (model.features[0].weight.view(-1).square().sum().detach().cpu().numpy())
        norm1_list[x] = (model.features[0].weight[0,0,0,0].detach().cpu().numpy())

        accuracy = running_correct.double()/(len(trainloader.dataset))
        print('epoch %d loss %.6f accuracy %.6f' %(x, running_loss/(batch_id), accuracy))
        #writer.add_scalar('Loss/train', running_loss/batch_id, x)   ## loss/#batches 
        if ((x)%num_acc == 0) or (x==cfg.epochs-1):
            acc_list[y] = test(testloader, model, device)
            y = y+1
        if x%200 == 0:
            model_path = arch + '_' + dataset  + '_p_'+ str(precision) + '_model_' + str(checkpoint_epoch+x)+ '.pth'
            torch.save({'epoch': (checkpoint_epoch+x), 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'loss': running_loss/batch_id, 'accuracy': accuracy}, model_path)
                #utils.collect_gradients(params, faulty_layers)
    np.savetxt("outputs/first_pt_rmvd_noise_50/norm.txt", norm_list)
    np.savetxt("outputs/first_pt_rmvd_noise_50/norm_comp.txt", norm1_list)
    np.savetxt("outputs/first_pt_rmvd_noise_50/test_acc.txt", acc_list)
    np.savetxt("outputs/first_pt_rmvd_noise_50/loss.txt", loss_list)
           
def test(testloader, model, device):            
    model.eval()
    running_correct = 0.0

    with torch.no_grad():

      for t, (inputs,classes) in enumerate(testloader):

          inputs = inputs.to(device)
          classes = classes.to(device)
          model_outputs =model(inputs)
          #pdb.set_trace()
          lg, preds = torch.max(model_outputs, 1)
          correct=torch.sum(preds == classes.data)
          running_correct += correct
          
    acc = (running_correct.double()/(len(testloader.dataset)))
    print('Eval Accuracy %.3f'%(acc))
    return acc



