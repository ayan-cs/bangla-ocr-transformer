import numpy as np
import os, string, unicodedata, gc, torch, torchvision, editdistance, pickle
from itertools import groupby
import torchvision.transforms as T
from pathlib import Path

from services import DataGenerator
from model import BanglaOCR
from utils import load_function

def test(model, test_loader, tokenizer, max_text_length, device):
    model.eval()
    predicts = []
    gt = []
    imgs = []
    total = len(test_loader)
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            #print(f"Batch {idx} started", end='\t')
            """if idx % 50 == 0 :
                print(f"{idx} images processed out of {total} images")"""
            if idx == 10 :
                break
            src, trg = batch
            imgs.append(src.flatten(0,1))
            src, trg = src.to(device), trg.to(device)            
            memory = get_memory(model,src.float())
            out_indexes = [tokenizer.chars.index('SOS'), ]
            for i in range(max_text_length):
                mask = model.generate_square_subsequent_mask(i+1).to(device)
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model.vocab(model.tf_decoder(model.query_pos(model.decoder(trg_tensor)), memory, tgt_mask=mask))
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == tokenizer.chars.index('EOS'):
                    break
            predicts.append(tokenizer.decode(out_indexes))
            gt.append(tokenizer.decode(trg.flatten(0,1)))
            del src
            del trg
            gc.collect()
            torch.cuda.empty_cache()
    return predicts, gt, imgs

def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        pd, gt = pd.lower(), gt.lower()

        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("utf-8", "ignore").decode("utf-8")
            gt = unicodedata.normalize("NFKD", gt).encode("utf-8", "ignore").decode("utf-8")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split(), gt.split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    metrics = [cer, wer, ser]
    metrics = np.mean(metrics, axis=1)
    return metrics

def get_memory(model, imgs):
    x = model.conv(model.get_feature(imgs))
    bs,_,H, W = x.shape
    pos = torch.cat([
            model.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            model.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
    return pos +  0.1 * x.flatten(2).permute(2, 0, 1)

def run_inference(model_name, output):
    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]

    tokenizer_path = os.path.join(parent, 'Outputs', 'tokenizer.pk')
    source_path_val = os.path.join(parent, 'data', 'Validation', 'Images')
    gt_path_val = os.path.join(parent, 'data', 'Validation', 'Annotations')
    checkpoint_path = os.path.join(parent, 'Saved Checkpoints')

    print(f"Using model : {model_name}")
    output.write(f"Using model : {model_name}\n")
    properties = model_name.split('_')
    properties[-1] = properties[-1].rsplit('.', maxsplit=1)[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device type : {device}")
    output.write(f"Device type : {device}\n")

    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    charset_base = tokenizer.chars[4:]
    d_model, nheads, num_decoder_layers = int(properties[2]), int(properties[3]), int(properties[4])
    model = BanglaOCR(
        vocab_len = tokenizer.vocab_size,
        max_text_length = tokenizer.maxlen,
        hidden_dim = d_model,
        nheads = nheads,
        num_decoder_layers = num_decoder_layers
    )
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{model_name}.pth')))
    _ = model.to(device)

    transform = T.Compose([T.ToTensor()])
    test_loader = torch.utils.data.DataLoader(DataGenerator(source_path_val, gt_path_val, charset_base, tokenizer.maxlen, transform), batch_size = 1, shuffle = True)

    predicts, gt, imgs = test(model, test_loader, tokenizer, tokenizer.maxlen, device)
    predicts = list(map(lambda x : x.replace('SOS','').replace('EOS',''),predicts))
    gt = list(map(lambda x : x.replace('SOS','').replace('EOS',''),gt))
    evaluate = ocr_metrics(predicts = predicts, ground_truth = gt,)
    print(f"Character Error Rate {evaluate[0]}, Word Error Rate {evaluate[1]} and Sequence Error Rate {evaluate[2]}")
    output.write(f"Character Error Rate {evaluate[0]}, Word Error Rate {evaluate[1]} and Sequence Error Rate {evaluate[2]}\n")

    for i, item in enumerate(imgs[:10]):
        print("=" * 50, "\n")
        #output.write("="*50, '\n')
        print(f"\nGround truth : {gt[i]}\n\nPrediction : {predicts[i]}\n")
        output.write(f"\nGround truth : {gt[i]}\n\nPrediction : {predicts[i]}\n\n")