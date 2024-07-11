import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
import matplotlib.pyplot as plt 


def accuracy_fn(y_true:torch.Tensor,y_pred:torch.Tensor) -> float:
    correct =  y_true.eq(y_pred).sum().item()
    return (correct / len(y_true)) * 100

def train_step(module:nn.Module,data_loader:DataLoader,loss_fn = nn.Module,optimizer = torch.optim.Optimizer,accuracy_fn = None, device:torch.device = 'cpu',axes:Tuple=None,fig: plt.Figure = None)->Tuple:
    train_loss,train_acc,epoch_loss,epoch_acc = 0,0,[],[]
    for batch_idx, (X, y) in enumerate(data_loader):
        module.to(device)
        X,y = X.to(device), y.to(device)        
        y_pred = module(X)
        loss = loss_fn(y_pred,y)
        if hasattr(module,'l2_regularization'):
            reg = module.l2_regularization()
            loss += reg
        train_loss += loss
        epoch_loss.append(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if accuracy_fn:
            acc = accuracy_fn(y,y_pred.argmax(dim=1))
            train_acc += acc
            epoch_acc.append(acc)
        if axes and fig:
            #print(epoch_loss)
            loss_ax, acc_ax = axes
            from IPython.display import clear_output,display
            clear_output(wait=True) 
            loss_ax.plot(epoch_loss,color = "blue" )
            acc_ax.plot(epoch_acc,color = "blue")
            loss_ax.set_title(f'current epoch Loss total batch:{len(data_loader)},current batch:{batch_idx}')
            acc_ax.set_title('current Accuracy')
            display(fig)
        #print(f'train epoch total batch:{len(data_loader)},current batch:{batch_idx} current loss:{loss}')

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'train_loss:{train_loss}, trian:acc:{train_acc}')
    return train_loss,train_acc,epoch_loss,epoch_acc

def test_step(module:nn.Module,data_loader:DataLoader,loss_fn = nn.Module,accuracy_fn = None, device:torch.device = 'cpu',axes:Tuple=None,train_info:Tuple=None,fig: plt.Figure = None)->Tuple:
    test_loss, test_acc,epoch_loss,epoch_acc = 0, 0,[],[]
    with torch.inference_mode():
        module.to(device)
        module.eval()
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device),y.to(device)
            y_pred = module(X)
            #y_pred = y_pred.argmax(dim=1)
            loss = loss_fn(y_pred,y)
            epoch_loss.append(loss.detach().cpu().numpy())
            test_loss += loss
            if accuracy_fn:
                acc = accuracy_fn(y,y_pred.argmax(dim=1))
                epoch_acc.append(acc)
                test_acc += acc
            if axes and train_info and fig:
                loss_ax, acc_ax = axes
                from IPython.display import clear_output,display
                clear_output(wait=True) 
                train_losses, train_accs = train_info
                loss_ax.plot(train_losses,color = "blue" )
                loss_ax.plot(epoch_loss,color = "orange" )
                acc_ax.plot(train_accs,color = "blue")
                acc_ax.plot(epoch_acc,color = "orange")
                loss_ax.set_title(f'current epoch Loss total batch:{len(data_loader)},current batch:{batch_idx}')
                acc_ax.set_title('current Accuracy')
                display(fig)
            #print(f'test epoch total batch:{len(data_loader)},current batch:{batch_idx} current loss:{loss}')
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    print(f'test loss {test_loss}, test acc:{test_acc}')
    return test_loss,test_acc