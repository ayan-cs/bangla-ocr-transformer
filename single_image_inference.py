import os, sys, mlconfig, pickle, torch, cv2
from model import BanglaOCR
import torchvision.transforms as T
from pathlib import Path
import numpy as np

def preprocess(img_path : str):
    
    def resize(img):
        h, w = img.shape[0], img.shape[1] # (h, w)
        new_height = int(np.ceil((w * 16)/9))
        if h > new_height :
            new_width = int(np.ceil((h * 9)/16))
            new_img = np.full((h, new_width), 255, dtype=np.uint8) # In case of np.full -> (h, w)
            new_img[:, :w] = img # (h, w)
        else:
            new_img = np.full((new_height, w), 255, dtype=np.uint8)
            new_img[:h, :] = img # (h, w)
        return cv2.resize(new_img, (360, 640)) # (720, 1280) or (360, 640) -> (w, h)
    
    def binarize(image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        return binary
    
    image = cv2.imread(img_path, 0)
    image = binarize(image)
    image = resize(image)
    image = np.repeat(image[..., np.newaxis],3, -1)
    return image

def get_memory(model, imgs):
    print("Inside get memory")
    x = model.conv(model.get_feature(imgs))
    print("Obatined x")
    bs,_,H, W = x.shape
    pos = torch.cat([
            model.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            model.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
    return pos +  0.1 * x.flatten(2).permute(2, 0, 1)

def single_image_inference(model, img, tokenizer, transform, device):
    '''
    Run inference on single image
    '''
    img = transform(img)    
    imgs = img.unsqueeze(0).to(device)
    print("Before for loop")
    with torch.no_grad():
      memory = get_memory(model, imgs.float())
      print("Obtained memory")
      out_indexes = [tokenizer.chars.index('SOS'), ]
      print("Starting")
      for i in range(tokenizer.maxlen):
            mask = model.generate_square_subsequent_mask(i+1).to(device)
            trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
            output = model.vocab(model.tf_decoder(model.query_pos(model.decoder(trg_tensor)), memory,tgt_mask=mask))
            out_token = output.argmax(2)[-1].item()
            if out_token == tokenizer.chars.index('EOS'):
                break
            out_indexes.append(out_token)

    pre = tokenizer.decode(out_indexes[1:])
    return pre

config = 'inference_config.yaml'
parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]
config = mlconfig.load(os.path.join(parent, config))
config.set_immutable()

image_path = os.path.join(parent, 'data', config.image)
checkpoint_path = os.path.join(parent, 'Saved Checkpoints', f"{config.model_name}.pth")
tokenizer_path = os.path.join(parent, 'Outputs', 'tokenizer.pk')

if not os.path.exists(image_path):
    sys.exit(f"Image file {config.image} does not exist!")
if not os.path.exists(checkpoint_path):
    sys.exit(f"Model {config.model_name} does not exist!")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model_name = config.model_name
properties = model_name.split('_')
d_model, nheads, num_decoder_layers = int(properties[2]), int(properties[3]), int(properties[4])
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

model = BanglaOCR(
            vocab_len = tokenizer.vocab_size,
            max_text_length = tokenizer.maxlen,
            hidden_dim = d_model,
            nheads = nheads,
            num_decoder_layers = num_decoder_layers
        )
model.load_state_dict(torch.load(os.path.join(checkpoint_path)))
model = model.to(device)

transform = T.Compose([T.ToTensor()])

image = preprocess(image_path)
print(f"Preprocessing complete {image.shape}")
predicted = single_image_inference(model, image, tokenizer, transform, device)
print(f"Predicted text :\n\n{predicted}")