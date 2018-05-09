import pickle
import torch

def save_checkpoint(file_name, model, optimizer, epochs_finished, other_info={}):

    checkpoint = {
        'model' :  model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epochs_finished': epochs_finished,
        'other_info' : other_info # Additional Parameter Dict
    }

    torch.save(checkpoint, file_name)

def restore_checkpoint(file_name, model, optimizer):
    checkpoint = torch.load(file_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    current_epoch = checkpoint['epochs_finished'] + 1
    other_info = checkpoint['other_info']
    return model, optimizer, current_epoch, other_info
    
'''def restore_checkpoint(file_name, model, optimizer):
    f=open(file_name, 'rb')  
    checkpoint=pickle.load(f)  
    f.close() 

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    current_epoch = checkpoint['epochs_finished'] + 1
    other_info = checkpoint['other_info']
    
    return model, optimizer, current_epoch, other_info

def save_checkpoint(file_name, model, optimizer, epochs_finished, other_info={}):
    checkpoint = {
        'model' :  model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epochs_finished': epochs_finished,
        'other_info' : other_info # Additional Parameter Dict
    }
    f = open(file_name, "wb")
    pickle.dump(checkpoint, f)
    f.close()
    print("saved checkpoint")'''