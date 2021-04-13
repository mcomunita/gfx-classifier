import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import utils

def train_fx_net(model, optimizer, train_loader, train_sampler, epoch, loss_function=nn.CrossEntropyLoss(), device='cpu'):
    model.train()
    train_set_size = len(train_sampler)
    total_loss = 0
    total_correct = 0
    results = []

    for batch_idx, data in enumerate(train_loader):
        mels, labels, settings, filenames, indeces = data
        
        mels = mels.to(device)
        labels = labels.to(device)
        
        preds = model(mels) # pass batch
        # loss = F.cross_entropy(preds, labels) # calculate loss
        loss = loss_function(preds, labels)

        optimizer.zero_grad() # zero gradients otherwise get accumulated
        loss.backward() # calculate gradient
        optimizer.step() # update weights

        total_loss += loss.item()
        correct = utils.get_num_correct_labels(preds, labels)
        total_correct += correct
        
        for idx, filename in enumerate(filenames):
            results.append(
                (indeces[idx].item(), 
                 filename, 
                 preds[idx].argmax().item(),
                 labels[idx].item()))
        
        if batch_idx > 0 and batch_idx % 50 == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tAvg Loss: {:.4f}'.format(
                        epoch, # epoch
                        batch_idx * len(labels), 
                        train_set_size,
                        100. * batch_idx / len(train_loader), # % completion
                        total_loss,
                        total_loss / (batch_idx * len(labels))))

    print('====> Epoch: {}\tTotal Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{}\tPercentage Correct: {:.2f}'.format(
            epoch,
            total_loss,
            total_loss / train_set_size,
            total_correct,
            train_set_size,
            100 * total_correct / train_set_size))
    
    return total_loss, total_correct, results


def val_fx_net(model, val_loader, val_sampler, loss_function=nn.CrossEntropyLoss(), device='cpu'):
    model.eval()
    val_set_size = len(val_sampler)
    total_loss = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            mels, labels, settings, filenames, indeces = data

            mels = mels.to(device)
            labels = labels.to(device)

            preds = model(mels) # Pass Batch
            # loss = F.cross_entropy(preds, labels) # Calculate Loss
            loss = loss_function(preds, labels)

            total_loss += loss.item()
            correct = utils.get_num_correct_labels(preds, labels)
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                     filename, 
                     preds.argmax(dim=1)[idx].item(), 
                     labels[idx].item()))
    
    print('====> Val Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{}\tPercentage Correct: {:.2f}'.format(
        total_loss,
        total_loss / val_set_size,
        total_correct,
        val_set_size,
        100 * total_correct / val_set_size))
    
    return total_loss, total_correct, results
    

def test_fx_net(model, test_loader, test_sampler, loss_function=nn.CrossEntropyLoss(), device='cpu'):
    model.eval()
    test_set_size = len(test_sampler)
    total_loss = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            mels, labels, settings, filenames, indeces = data
            
            mels = mels.to(device)
            labels = labels.to(device)

            preds = model(mels) # Pass Batch
            # loss = F.cross_entropy(preds, labels) # Calculate Loss
            loss = loss_function(preds, labels)

            total_loss += loss.item()
            correct = utils.get_num_correct_labels(preds, labels)
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                     filename, 
                     preds.argmax(dim=1)[idx].item(), 
                     labels[idx].item()))

    print('====> Test Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{}\tPercentage Correct: {:.2f}'.format(
        total_loss,
        total_loss / test_set_size,
        total_correct,
        test_set_size,
        100 * total_correct / test_set_size))
    
    return total_loss, total_correct, results


# ===============================================================
def train_settings_net(model, optimizer, train_loader, train_sampler, epoch, loss_function=nn.L1Loss(), device='cpu'):
    model.train()
    train_set_size = len(train_sampler)
    total_loss = 0
    total_correct = 0
    results = []

    for batch_idx, data in enumerate(train_loader): # Get Batch
        mels, labels, settings, filenames, indeces = data
        
        preds = model(mels)
        
        loss = loss_function(preds, settings)

        optimizer.zero_grad() # Zero gradients otherwise get accumulated
        loss.backward() # Calculate Gradient
        optimizer.step() # Update Weights

        total_loss += loss.item()
        correct = utils.get_num_correct_settings(preds, settings)
        total_correct += correct
        
        for idx, filename in enumerate(filenames):
            results.append(
                (indeces[idx].item(), 
                 filename, 
                 np.round(preds[idx].detach().numpy(), 3),
                 np.round(settings[idx].detach().numpy(), 3)))
        
        if batch_idx > 0 and batch_idx % 50 == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tAvg Loss: {:.4f}'.format(
                        epoch, # epoch
                        batch_idx * len(labels), 
                        train_set_size,
                        100. * batch_idx / len(train_loader), # % completion
                        total_loss,
                        total_loss / (batch_idx * len(labels))))

    print('====> Epoch: {}\tTotal Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{:.0f}\tPercentage Correct: {:.2f}'.format(
            epoch,
            total_loss,
            total_loss / train_set_size,
            total_correct,
            train_set_size,
            100 * total_correct / train_set_size))
    
    return total_loss, total_correct, results

        
def val_settings_net(model, val_loader, val_sampler, loss_function=nn.L1Loss(), device='cpu'):
    model.eval()
    val_set_size = len(val_sampler)
    total_loss = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader): # Get Batch
            mels, labels, settings, filenames, indeces = data
            
            preds = model(mels)
            
            loss = loss_function(preds, settings)

            total_loss += loss.item()
            correct = utils.get_num_correct_settings(preds, settings)
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    np.round(preds[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
        
    print('====> Val Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{:.0f}\tPercentage Correct: {:.2f}'.format(
        total_loss,
        total_loss / val_set_size,
        total_correct,
        val_set_size,
        100 * total_correct / val_set_size))
    
    return total_loss, total_correct, results

    
def test_settings_net(model, test_loader, test_sampler, loss_function=nn.L1Loss(), device='cpu'):
    model.eval()
    test_set_size = len(test_sampler)
    total_loss = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            mels, labels, settings, filenames, indeces = data
            
            preds = model(mels)
            
            loss = loss_function(preds, settings)

            total_loss += loss.item()
            correct = utils.get_num_correct_settings(preds, settings)
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    np.round(preds[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
        
    print('====> Test Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{}\tPercentage Correct: {:.2f}'.format(
        total_loss,
        total_loss / test_set_size,
        total_correct,
        test_set_size,
        100 * total_correct / test_set_size))
    
    return total_loss, total_correct, results


# ===============================================================
def train_settings_cond_net(model, optimizer, train_loader, train_sampler, epoch, loss_function=nn.L1Loss(), device='cpu'):
    model.train()
    train_set_size = len(train_sampler)
    total_loss = 0
    total_correct = 0
    results = []

    for batch_idx, data in enumerate(train_loader):
        mels, labels, settings, filenames, indeces = data
        mels = mels.unsqueeze(1)
        
        mels = mels.to(device)
        labels = labels.to(device)
        settings = settings.to(device)
        
        # predictions, loss and gradient
        preds = model(mels, labels)  # pass batch and labels for conditioning
        loss = loss_function(preds, settings)  # calculate loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct = utils.get_num_correct_settings(preds, settings)
        total_correct += correct
    
        for idx, filename in enumerate(filenames):
            results.append(
                (indeces[idx].item(), 
                 filename, 
                 np.round(preds[idx].detach().numpy(), 3),
                 np.round(settings[idx].detach().numpy(), 3)))
        
        if batch_idx > 0 and batch_idx % 50 == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tAvg Loss: {:.4f}'.format(
                        epoch,
                        batch_idx * len(labels), 
                        train_set_size,
                        100. * batch_idx / len(train_loader), # % completion
                        total_loss,
                        total_loss / (batch_idx * len(labels))))

    print('====> Epoch: {}\tTotal Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{:.0f}\tPercentage Correct: {:.2f}'.format(
            epoch,
            total_loss,
            total_loss / train_set_size,
            total_correct,
            train_set_size,
            100 * total_correct / train_set_size))
    
    return total_loss, total_correct, results


def val_settings_cond_net(model, val_loader, val_sampler, loss_function=nn.L1Loss(), device='cpu'):
    model.eval()
    val_set_size = len(val_sampler)
    total_loss = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader): # Get Batch
            mels, labels, settings, filenames, indeces = data
            mels = mels.unsqueeze(1)
            
            mels = mels.to(device)
            labels = labels.to(device)
            settings = settings.to(device)

            # predictions and loss
            preds = model(mels, labels)  # pass batch and labels for conditioning
            loss = loss_function(preds, settings)  # calculate loss

            total_loss += loss.item()
            correct = utils.get_num_correct_settings(preds, settings)
            total_correct += correct

            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    np.round(preds[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
    
    print('====> Val Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{:.0f}\tPercentage Correct: {:.2f}'.format(
        total_loss,
        total_loss / val_set_size,
        total_correct,
        val_set_size,
        100 * total_correct / val_set_size))
    
    return total_loss, total_correct, results


def test_settings_cond_net(model, test_loader, test_sampler, loss_function=nn.L1Loss(), device='cpu'):
    model.eval()
    test_set_size = len(test_sampler)
    total_loss = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            mels, labels, settings, filenames, indeces = data
            mels = mels.unsqueeze(1)

            mels = mels.to(device)
            labels = labels.to(device)
            settings = settings.to(device)

            # predictions and loss
            preds = model(mels, labels)  # pass batch and labels for conditioning
            loss = loss_function(preds, settings)  # calculate loss

            total_loss += loss.item()
            correct = utils.get_num_correct_settings(preds, settings)
            total_correct += correct

            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    np.round(preds[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
        
    print('====> Test Loss: {:.4f}\t Avg Loss: {:.4f}\tCorrect: {:.0f}/{}\tPercentage Correct: {:.2f}'.format(
        total_loss,
        total_loss / test_set_size,
        total_correct,
        test_set_size,
        100 * total_correct / test_set_size))
    
    return total_loss, total_correct, results


# ===============================================================
def train_multi_net(model, optimizer, train_loader, train_sampler, epoch, 
                    loss_function_fx=nn.CrossEntropyLoss(), loss_function_set=nn.L1Loss(), device='cpu'):
    model.train()
    train_set_size = len(train_sampler)
    total_loss_fx = 0
    total_loss_set = 0
    total_loss = 0
    total_correct_fx = 0
    total_correct_set = 0
    total_correct = 0
    results = []

    for batch_idx, data in enumerate(train_loader):
        mels, labels, settings, filenames, indeces = data
        
        mels = mels.to(device)
        labels = labels.to(device)
        settings = settings.to(device)
        
        preds_fx, preds_set = model(mels)

        loss_fx = loss_function_fx(preds_fx, labels)
        loss_set = loss_function_set(preds_set, settings)
        loss = loss_fx + loss_set

        optimizer.zero_grad() # zero gradients otherwise get accumulated
        loss.backward() # calculate gradient
        optimizer.step() # update weights

        total_loss_fx += loss_fx.item()
        total_loss_set += loss_set.item()
        total_loss += loss.item()
        correct_fx = utils.get_num_correct_labels(preds_fx, labels)
        correct_set = utils.get_num_correct_settings(preds_set, settings)
        correct = utils.get_num_correct(preds_fx, preds_set, labels, settings)
        total_correct_fx += correct_fx
        total_correct_set += correct_set
        total_correct += correct
        
        for idx, filename in enumerate(filenames):
            results.append(
                (indeces[idx].item(), 
                filename, 
                preds_fx[idx].argmax().item(),
                labels[idx].item(),
                np.round(preds_set[idx].detach().numpy(), 3),
                np.round(settings[idx].detach().numpy(), 3)))
        
        if batch_idx > 0 and batch_idx % 50 == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tAvg Loss: {:.4f}'.format(
                        epoch, # epoch
                        batch_idx * len(labels), 
                        train_set_size,
                        100. * batch_idx / len(train_loader), # % completion
                        total_loss,
                        total_loss / (batch_idx * len(labels))))

    print('====> Epoch: {}'
                        '\tTotal Loss: {:.4f}'
                        '\t Avg Loss: {:.4f}'
                        '\t Fx Loss: {:.4f}'
                        '\t Set Loss: {:.4f}'
                        '\n\t\tCorrect: {:.0f}/{}'
                        '\tFx Correct: {:.0f}/{}'
                        '\tSet Correct: {:.0f}/{}'
                        '\n\t\tPercentage Correct: {:.2f}'
                        '\tPercentage Fx Correct: {:.2f}'
                        '\tPercentage Set Correct: {:.2f}'.format(
                            epoch,
                            total_loss,
                            total_loss / train_set_size,
                            total_loss_fx,
                            total_loss_set,
                            total_correct,
                            train_set_size,
                            total_correct_fx,
                            train_set_size,
                            total_correct_set,
                            train_set_size,
                            100 * total_correct / train_set_size,
                            100 * total_correct_fx / train_set_size,
                            100 * total_correct_set / train_set_size))
    
    return total_loss, total_correct, results


def val_multi_net(model, val_loader, val_sampler, loss_function_fx=nn.CrossEntropyLoss(), loss_function_set=nn.L1Loss(), device='cpu'):
    model.eval()
    val_set_size = len(val_sampler)
    total_loss_fx = 0
    total_loss_set = 0
    total_loss = 0
    total_correct_fx = 0
    total_correct_set = 0
    total_correct = 0
    results = []

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            mels, labels, settings, filenames, indeces = data
            
            mels = mels.to(device)
            labels = labels.to(device)
            settings = settings.to(device)
            
            preds_fx, preds_set = model(mels)
            loss_fx = loss_function_fx(preds_fx, labels)
            loss_set = loss_function_set(preds_set, settings)
            loss = loss_fx + loss_set

            total_loss_fx += loss_fx.item()
            total_loss_set += loss_set.item()
            total_loss += loss.item()
            correct_fx = utils.get_num_correct_labels(preds_fx, labels)
            correct_set = utils.get_num_correct_settings(preds_set, settings)
            correct = utils.get_num_correct(preds_fx, preds_set, labels, settings)
            total_correct_fx += correct_fx
            total_correct_set += correct_set
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    preds_fx[idx].argmax().item(),
                    labels[idx].item(),
                    np.round(preds_set[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
    
    print('====> Val Loss: {:.4f}'
                '\t Avg Loss: {:.4f}'
                '\t Fx Loss: {:.4f}'
                '\t Set Loss: {:.4f}'
                '\n\t\tCorrect: {:.0f}/{:.0f}'
                '\tFx Correct: {:.0f}/{}'
                '\tSet Correct: {:.0f}/{}'
                '\n\t\tPercentage Correct: {:.2f}'
                '\tPercentage Fx Correct: {:.2f}'
                '\tPercentage Set Correct: {:.2f}'.format(
                    total_loss,
                    total_loss / val_set_size,
                    total_loss_fx,
                    total_loss_set,
                    total_correct,
                    val_set_size,
                    total_correct_fx,
                    val_set_size,
                    total_correct_set,
                    val_set_size,
                    100 * total_correct / val_set_size,
                    100 * total_correct_fx / val_set_size,
                    100 * total_correct_set / val_set_size))
    
    return total_loss, total_correct, results


def test_multi_net(model, test_loader, test_sampler, loss_function_fx=nn.CrossEntropyLoss(), loss_function_set=nn.L1Loss(), device='cpu'):
    model.eval()
    test_set_size = len(test_sampler)
    total_loss_fx = 0
    total_loss_set = 0
    total_loss = 0
    total_correct_fx = 0
    total_correct_set = 0
    total_correct = 0
    results = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            mels, labels, settings, filenames, indeces = data
            
            mels = mels.to(device)
            labels = labels.to(device)
            settings = settings.to(device)
            
            preds_fx, preds_set = model(mels)
            loss_fx = loss_function_fx(preds_fx, labels)
            loss_set = loss_function_set(preds_set, settings)
            loss = loss_fx + loss_set

            total_loss_fx += loss_fx.item()
            total_loss_set += loss_set.item()
            total_loss += loss.item()
            correct_fx = utils.get_num_correct_labels(preds_fx, labels)
            correct_set = utils.get_num_correct_settings(preds_set, settings)
            correct = utils.get_num_correct(preds_fx, preds_set, labels, settings)
            total_correct_fx += correct_fx
            total_correct_set += correct_set
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    preds_fx[idx].argmax().item(),
                    labels[idx].item(),
                    np.round(preds_set[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
    
    print('====> Test Loss: {:.4f}'
                '\t Avg Loss: {:.4f}'
                '\t Fx Loss: {:.4f}'
                '\t Set Loss: {:.4f}'
                '\n\t\tCorrect: {:.0f}/{:.0f}'
                '\tFx Correct: {:.0f}/{}'
                '\tSet Correct: {:.0f}/{}'
                '\n\t\tPercentage Correct: {:.2f}'
                '\tPercentage Fx Correct: {:.2f}'
                '\tPercentage Set Correct: {:.2f}'.format(
                    total_loss,
                    total_loss / test_set_size,
                    total_loss_fx,
                    total_loss_set,
                    total_correct,
                    test_set_size,
                    total_correct_fx,
                    test_set_size,
                    total_correct_set,
                    test_set_size,
                    100 * total_correct / test_set_size,
                    100 * total_correct_fx / test_set_size,
                    100 * total_correct_set / test_set_size))
    
    return total_loss, total_correct, results



# ===============================================================
def train_cond_nets(model_fx, model_set, 
                    optimizer_fx, optimizer_set, 
                    train_loader, train_sampler, epoch,
                    loss_function_fx=nn.CrossEntropyLoss(), loss_function_set=nn.L1Loss(), 
                    device='cpu'):
    model_fx.train()
    model_set.train()
    train_set_size = len(train_sampler)
    total_loss_fx = 0
    total_loss_set = 0
    total_loss = 0
    total_correct_fx = 0
    total_correct_set = 0
    total_correct = 0
    results = []

    for batch_idx, data in enumerate(train_loader):
        mels, labels, settings, filenames, indeces = data
        
        mels = mels.to(device)
        labels = labels.to(device)
        settings = settings.to(device)
        
        # predictions, loss and gradient for FxNet 
        preds_fx = model_fx(mels)
        loss_fx = loss_function_fx(preds_fx, labels)

        optimizer_fx.zero_grad()
        loss_fx.backward()
        optimizer_fx.step()

        total_loss_fx += loss_fx.item()
        correct_fx = utils.get_num_correct_labels(preds_fx, labels)
        total_correct_fx += correct_fx
        
        # predictions, loss and gradient for SettingsNet
        cond_set = preds_fx.argmax(dim=1) # calculate labels for conditioning of setnet
        preds_set = model_set(mels, cond_set)  # pass batch and labels for conditioning
        loss_set = loss_function_set(preds_set, settings)  # calculate loss

        optimizer_set.zero_grad()
        loss_set.backward()
        optimizer_set.step()

        total_loss_set += loss_set.item()
        correct_set = utils.get_num_correct_settings(preds_set, settings)
        total_correct_set += correct_set

        # predictions and loss for both networks
        loss = loss_fx + loss_set
        total_loss += loss.item()

        correct = utils.get_num_correct(preds_fx, preds_set, labels, settings)
        total_correct += correct
        
        for idx, filename in enumerate(filenames):
            results.append(
                (indeces[idx].item(), 
                filename, 
                preds_fx[idx].argmax().item(),
                labels[idx].item(),
                np.round(preds_set[idx].detach().numpy(), 3),
                np.round(settings[idx].detach().numpy(), 3)))
    
        if batch_idx > 0 and batch_idx % 50 == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tTotal Loss: {:.4f}\tAvg Loss: {:.4f}'.format(
                        epoch, # epoch
                        batch_idx * len(labels), 
                        train_set_size,
                        100. * batch_idx / len(train_loader), # % completion
                        total_loss,
                        total_loss / (batch_idx * len(labels))))

    print('====> Epoch: {}'
                        '\tTotal Loss: {:.4f}'
                        '\t Avg Loss: {:.4f}'
                        '\t Fx Loss: {:.4f}'
                        '\t Set Loss: {:.4f}'
                        '\n\t\tCorrect: {:.0f}/{}'
                        '\tFx Correct: {:.0f}/{}'
                        '\tSet Correct: {:.0f}/{}'
                        '\n\t\tPercentage Correct: {:.2f}'
                        '\tPercentage Fx Correct: {:.2f}'
                        '\tPercentage Set Correct: {:.2f}'.format(
                            epoch,
                            total_loss,
                            total_loss / train_set_size,
                            total_loss_fx,
                            total_loss_set,
                            total_correct,
                            train_set_size,
                            total_correct_fx,
                            train_set_size,
                            total_correct_set,
                            train_set_size,
                            100 * total_correct / train_set_size,
                            100 * total_correct_fx / train_set_size,
                            100 * total_correct_set / train_set_size))
    
    return total_loss, total_correct, results


def val_cond_nets(model_fx, model_set, 
                    val_loader, val_sampler,
                    loss_function_fx=nn.CrossEntropyLoss(), loss_function_set=nn.L1Loss(), 
                    device='cpu'):
    model_fx.eval()
    model_set.eval()
    val_set_size = len(val_sampler)
    total_loss_fx = 0
    total_loss_set = 0
    total_loss = 0
    total_correct_fx = 0
    total_correct_set = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader): # Get Batch
            mels, labels, settings, filenames, indeces = data
            
            mels = mels.to(device)
            labels = labels.to(device)
            settings = settings.to(device)

            # predictions and loss for FxNet 
            preds_fx = model_fx(mels)
            loss_fx = loss_function_fx(preds_fx, labels)

            total_loss_fx += loss_fx.item()
            correct_fx = utils.get_num_correct_labels(preds_fx, labels)
            total_correct_fx += correct_fx

            # predictions and loss SettingsNet
            cond_set = preds_fx.argmax(dim=1) # calculate labels for conditioning of setnet 
            preds_set = model_set(mels, cond_set)  # pass batch and labels for conditioning
            loss_set = loss_function_set(preds_set, settings)  # calculate loss

            total_loss_set += loss_set.item()
            correct_set = utils.get_num_correct_settings(preds_set, settings)
            total_correct_set += correct_set

            # predictions and loss for both networks
            loss = loss_fx + loss_set
            total_loss += loss.item()

            correct = utils.get_num_correct(preds_fx, preds_set, labels, settings)
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    preds_fx[idx].argmax().item(),
                    labels[idx].item(),
                    np.round(preds_set[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
    
    print('====> Val Loss: {:.4f}'
                '\t Avg Loss: {:.4f}'
                '\t Fx Loss: {:.4f}'
                '\t Set Loss: {:.4f}'
                '\n\t\tCorrect: {:.0f}/{:.0f}'
                '\tFx Correct: {:.0f}/{}'
                '\tSet Correct: {:.0f}/{}'
                '\n\t\tPercentage Correct: {:.2f}'
                '\tPercentage Fx Correct: {:.2f}'
                '\tPercentage Set Correct: {:.2f}'.format(
                    total_loss,
                    total_loss / val_set_size,
                    total_loss_fx,
                    total_loss_set,
                    total_correct,
                    val_set_size,
                    total_correct_fx,
                    val_set_size,
                    total_correct_set,
                    val_set_size,
                    100 * total_correct / val_set_size,
                    100 * total_correct_fx / val_set_size,
                    100 * total_correct_set / val_set_size))
    
    return total_loss, total_correct, results


def test_cond_nets(model_fx, model_set, 
                    test_loader, test_sampler,
                    loss_function_fx=nn.CrossEntropyLoss(), loss_function_set=nn.L1Loss(), 
                    device='cpu'):
    model_fx.eval()
    model_set.eval()
    test_set_size = len(test_sampler)
    total_loss_fx = 0
    total_loss_set = 0
    total_loss = 0
    total_correct_fx = 0
    total_correct_set = 0
    total_correct = 0
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            mels, labels, settings, filenames, indeces = data

            mels = mels.to(device)
            labels = labels.to(device)
            settings = settings.to(device)

            # predictions and loss for FxNet 
            preds_fx = model_fx(mels)
            loss_fx = loss_function_fx(preds_fx, labels)

            total_loss_fx += loss_fx.item()
            correct_fx = utils.get_num_correct_labels(preds_fx, labels)
            total_correct_fx += correct_fx

            # predictions and loss for SettingsNet
            cond_set = preds_fx.argmax(dim=1) # calculate labels for conditioning of setnet 
            preds_set = model_set(mels, cond_set)  # pass batch and labels for conditioning
            loss_set = loss_function_set(preds_set, settings)  # calculate loss

            total_loss_set += loss_set.item()
            correct_set = utils.get_num_correct_settings(preds_set, settings)
            total_correct_set += correct_set

            # loss for both networks
            loss = loss_fx + loss_set
            total_loss += loss.item()

            correct = utils.get_num_correct(preds_fx, preds_set, labels, settings)
            total_correct += correct
            
            for idx, filename in enumerate(filenames):
                results.append(
                    (indeces[idx].item(), 
                    filename, 
                    preds_fx[idx].argmax().item(),
                    labels[idx].item(),
                    np.round(preds_set[idx].detach().numpy(), 3),
                    np.round(settings[idx].detach().numpy(), 3)))
    
    print('====> Test Loss: {:.4f}'
                '\t Avg Loss: {:.4f}'
                '\t Fx Loss: {:.4f}'
                '\t Set Loss: {:.4f}'
                '\n\t\tCorrect: {:.0f}/{:.0f}'
                '\tFx Correct: {:.0f}/{}'
                '\tSet Correct: {:.0f}/{}'
                '\n\t\tPercentage Correct: {:.2f}'
                '\tPercentage Fx Correct: {:.2f}'
                '\tPercentage Set Correct: {:.2f}'.format(
                    total_loss,
                    total_loss / test_set_size,
                    total_loss_fx,
                    total_loss_set,
                    total_correct,
                    test_set_size,
                    total_correct_fx,
                    test_set_size,
                    total_correct_set,
                    test_set_size,
                    100 * total_correct / test_set_size,
                    100 * total_correct_fx / test_set_size,
                    100 * total_correct_set / test_set_size))
    
    return total_loss, total_correct, results