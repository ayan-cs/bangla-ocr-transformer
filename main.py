import os, mlconfig, sys, shutil
from pathlib import Path
from train_model import trainer
from inference import run_inference

num_args = len(sys.argv)
preprocess = False
print(sys.argv)
if num_args > 2:
    sys.exit(f"Too many arguments : Expected atmost 1, got {num_args-1}")
if num_args < 2 :
    sys.exit(f"Expected atleast 1 argument\nUsage :\npython main.py <mode : train | inference>")
if num_args == 2:
    if sys.argv[1] not in ['train', 'inference']:
        sys.exit(f"Mode '{sys.argv[1]}' is invalid!")

mode = sys.argv[1]

if mode == 'train' :
    config = 'train_config.yaml'

    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]
    fn = os.path.join(parent, config)
    config = mlconfig.load(fn)
    config.set_immutable()

    trainer(config = config)

if mode == 'inference' :
    config = 'inference_config.yaml'

    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]
    fn = os.path.join(parent, config)
    config = mlconfig.load(fn)
    config.set_immutable()

    checkpoint_path = os.path.join(parent, 'Saved Checkpoints')
    models = [i for i in os.listdir(checkpoint_path) if i.endswith('.pth')]
    if len(models) < 1:
        sys.exit(f"No pre-trained models available. Train a model first.")
    if not os.path.exists(os.path.join(parent, 'data', 'Train')):
        sys.exit(f"Data directory not available!")
    model_name = f"{config.model_name}.pth"
    if model_name not in models:
        sys.exit(f"No model available with name : {model_name}")
    model_name = model_name.rsplit('.', maxsplit=1)[0]

    logs = os.path.join(parent, 'Outputs', f'inference_logs_{model_name}.txt')
    output = open(logs, 'w', encoding='utf-8')
    
    run_inference(model_name, output)
    output.close()