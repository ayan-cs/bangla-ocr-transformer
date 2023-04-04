import numpy as np
import datetime, os, torch, time, pickle, sys, gc
from torch import nn
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt

from services import Tokenizer, DataGenerator, LabelSmoothing
from model import BanglaOCR
from utils import findMaxTextLength, load_function

def train(model, criterion, optimiser, scheduler, dataloader, tokenizer):
    model.train()
    total_loss = 0
    #print("Training ...")
    batch = 0
    for imgs, labels_y in dataloader:
        #print(f"Batch : {batch}")
        imgs = imgs.to('cuda')
        labels_y = labels_y.to('cuda')

        optimiser.zero_grad()
        output = model(imgs.float(),labels_y.long()[:,:-1])

        norm = (labels_y != 0).sum()
        loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size), labels_y[:,1:].contiguous().view(-1).long()) / norm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        optimiser.step()
        total_loss += loss.item() * norm
        batch += 1
        del imgs
        del labels_y
        del output
        gc.collect()
        torch.cuda.empty_cache()
 
    return total_loss / len(dataloader)
 
def evaluate(model, criterion, dataloader, tokenizer):
    model.eval()
    epoch_loss = 0
    #print("Evaluation ...")
    batch = 0
    with torch.no_grad():
        for imgs, labels_y in dataloader:
            #print(f"Batch : {batch}")
            imgs = imgs.to('cuda')
            labels_y = labels_y.to('cuda')

            output = model(imgs.float(),labels_y.long()[:,:-1])
                
            norm = (labels_y != 0).sum()
            loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size), labels_y[:,1:].contiguous().view(-1).long()) / norm

            epoch_loss += loss.item() * norm
            batch += 1
            del imgs
            del labels_y
            del output
            gc.collect()
            torch.cuda.empty_cache()
 
    return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def generatePlots(train_loss_list, val_loss_list, path):
    if len(train_loss_list) == 0 or len(val_loss_list) == 0:
        print("List empty")
    else:
        plt.plot(range(len(train_loss_list)), train_loss_list, color='blue', label='Train Loss')
        plt.plot(range(len(val_loss_list)), val_loss_list, color='red', label='Valid loss')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Error Summary during training')
        plt.legend()
        plt.savefig(path)

def trainer(config):
    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]

    model_name = f"BanglaOCR_{config.epoch}_{config.model.d_model}_{config.model.nheads}_{config.model.num_decoder_layers}"

    source_path_train = os.path.join(parent, 'data', 'Train', 'Images')
    gt_path_train = os.path.join(parent, 'data', 'Train', 'Annotations')
    source_path_val = os.path.join(parent, 'data', 'Validation', 'Images')
    gt_path_val = os.path.join(parent, 'data', 'Validation', 'Annotations')
    checkpoint_path = os.path.join(parent, 'Saved Checkpoints', f'{model_name}.pth')
    tokenizer_path = os.path.join(parent, 'Outputs', 'tokenizer.pk')
    logs = os.path.join(parent, 'Outputs', f'train_logs_{model_name}.txt')
    plot_path = os.path.join(parent, 'Outputs', f'train_plots_{model_name}.jpg')
    charset_path = os.path.join(parent, 'charset', 'printable1.txt')
    punctuations_path = os.path.join(parent, 'charset', 'printable2.txt')

    if not os.path.exists(source_path_train):
        sys.exit(f"Data folder not available!")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    output = open(logs, 'w', encoding='utf-8')

    charset_base = []
    with open(charset_path, 'r', encoding='utf-8') as f:
        data = f.read()
        charset_base += data.split(',')
        charset_base = [x.strip() for x in charset_base]
    with open(punctuations_path, 'r', encoding='utf-8') as f:
        data = f.read()
        charset_base += data.split(' ')
        charset_base = [x.strip() for x in charset_base]
    charset_base.append(' ')

    input_size = (640, 360, 1)
    max_text_length = findMaxTextLength(gt_path_train, gt_path_val)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device type : {device}")
    output.write(f"Device type : {device}")

    transform = T.Compose([T.ToTensor()])
    tokenizer = Tokenizer(charset_base, max_text_length)
    pickle.dump(tokenizer, open(tokenizer_path, 'wb'))

    train_loader = torch.utils.data.DataLoader(DataGenerator(source_path_train, gt_path_train, charset_base, max_text_length, transform), batch_size = config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(DataGenerator(source_path_val, gt_path_val, charset_base, max_text_length, transform), batch_size = config.batch_size, shuffle=True)
    print(f"Number of batches in Train loader : {len(train_loader)}\nNumber of batches in Validation loader : {len(val_loader)}")
    output.write(f"Number of batches in Train loader : {len(train_loader)}\nNumber of batches in Validation loader : {len(val_loader)}")

    model = BanglaOCR(
        vocab_len = tokenizer.vocab_size,
        max_text_length = max_text_length,
        hidden_dim = config.model.d_model,
        nheads = config.model.nheads,
        num_decoder_layers = config.model.num_decoder_layers
    )
    _ = model.to(device)
    
    criterion = LabelSmoothing(size = tokenizer.vocab_size, padding_idx=0, smoothing=0.1)
    criterion.to(device)
    lr = .0001 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    best_valid_loss = np.inf
    c = 0
    train_loss_list = []
    val_loss_list = []

    print(f"Model '{model_name}' is being trained ...\nFor inference, put the model name in 'inference_config.yaml' file\n-> model_name : {model_name}\n")
    output.write(f"Model {model_name} is being trained ...\nFor inference, put the model name in 'inference_config.yaml' file\n-> model_name : {model_name}\n")
    for epoch in range(config.epoch):
        print(f'\nEpoch: {epoch+1:02}\tlearning rate : {scheduler.get_last_lr()}\n')
        output.write(f'\nEpoch: {epoch+1:02}\tlearning rate : {scheduler.get_last_lr()}\n')
        
        start_time = time.time()
    
        train_loss = train(model, criterion, optimizer, scheduler, train_loader, tokenizer)
        valid_loss = evaluate(model, criterion, val_loader, tokenizer)
        train_loss_list.append(train_loss)
        val_loss_list.append(valid_loss)
    
        epoch_mins, epoch_secs = epoch_time(start_time, time.time())
    
        c+=1
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            c=0
    
        if c>4:
            #decrease lr if loss does not deacrease after 5 steps
            scheduler.step()
            c=0
    
        print(f'Time: {epoch_mins}m {epoch_secs}s') 
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Val   Loss: {valid_loss:.3f}')
        output.write(f'Time: {epoch_mins}m {epoch_secs}s\nTrain Loss: {train_loss:.3f}\nVal   Loss: {valid_loss:.3f}')

    output.close()
    generatePlots(train_loss_list = train_loss_list, val_loss_list = val_loss_list, path = plot_path)