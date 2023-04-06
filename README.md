# BanglaOCR : Full Page Bangla handwritten text recognition using Image to Sequence Architecture

## Contents
Let's have a look at the brief explanations of what this repository folder/files contain and what their significance are.

- **charset** : Bangla, being an Indic language, I have used `utf-8` encoding for the representation purpose. The system uses **Character-level vocabulary**, and therefore, it is important to give our system a list of what the possible characters it is going to see. Thus, this directory contains two files :
    - `printable1.txt` : This file contains all possible (list NOT exhaustive) valid characters that can appear while writing Bangla.
    - `printable2.txt` : This file contains all possible (list NOT exhaustive) valid punctuations that can be used while writing any language including Bangla.
- **data** : This directory contains the data that we are going to use for Training the model as well as for the Validation purpose. For the directory structure, check out the next section.
- **Outputs** : This directory will contain *Tokenizer* file, *Training logs*, *Validation logs* and *Error plots* generated during training. You do not need to touch/modify this directory.
- **raw_data** : This will contain your raw data, which will be preprocessed before training/inference. You have to keep your dataset inside this directory. Data inside this directory can be preprocessed using `preprocess.ipynb` file. Throughout the process, raw images and ground truths will remain intact. As mentioned earlier, preprocessed images and ground truths will be stored inside **`data`** directory and from there, they will be used for training and inference.
- **Saved Checkpoints** : This directory will contain the trained models. The naming convention followed for naming the trained model is - `BanglaOCR_<No. of Epochs>_<Hidden layer dimension>_<No. of heads>_<No. of decoder layers>.pth`. For example, if a model name is `BanglaoCR_200_256_4_4.pth`, that means the model is trained for *200 epochs*, the model's *hidden layer dimension is 256*, it used *4 attention heads* and the number of *Transformer Decoder layer(s) is 4*. During Inference, you need to copy the model name from this directory **without extension** and put that within `inference_config.yaml` file. Moreover, **DO NOT change the model name** as the same model is loaded during inference by parsing the model name only.
- **`inference_config.yaml`** :
    - **`model_name`** : Put the model name (without extension), which you want to use during inference.
    - **`image`** : Required for `single_image_inference.py`. Put the image name here which you want to predict.
- **`inference.py`** : The main inference code is written here. This uses the `inference_config.yaml` file to read necessary details, reads preprocessed **validation** data from `data` directory and infers from the specified model accordingly. This is invoked by `main.py` when given command line argument as `inference`.
- **`main.py`** : The main driver code. You need to run this code in order to run the system. Details have been discussed in the upcoming section.
- **`model.py`** : The model/architecture class in contained here. Currently, the model uses `ResNet-18` as backbone/feature extractor and for both image embeddings and ground truth embeddings, **Sine** positional encodings are used. If you wish to change them, visit this file.
- **`preprocess.ipynb`** : This notebook is used to preprocess the data before training. This reads the raw data from **`raw_data`** directory and puts the preprocessed images inside **`data`** folder. Moreover, it splits the preprocessed data into **`Train`** and **`Validation`** split. Carefully go through the documentation inside to understand the functionalities and steps.
- **`services.py`** : This contains the `DataGenerator`, `Tokenizer` and `LabelSmoothing` classes.
    - **`DataGenerator`** creates data streaming from the `data` directory to the `DataLoader` during training and inference.
    - **`Tokenizer`** performs preprocessing the ground truths. The operations include text padding, generating numeric representation, creating and keeping records of vocabulary etc.
    - **`LabelSmoothing`** contains the loss function.
- **`single_image_infernece.py`** : As the name itself explains its functionality, it predicts/infers from a single image. All you need to do is to put that image inside the `data` folder and update the `inference_config.yaml` file accordingly. The code will do necessary preprocessing and generate its predictions. For details, visit the upcoming section.
- **`train_config.yaml`** : This contains necessary parameters required while training.
    - **epoch** : Specify the number of epochs.
    - **batch_size** : Specify the batch size accordint to your computing power and capacity.
    - **model.d_model** : Specify the size of the hidden layer.
    - **model.num_decoder_layers** : Specify the number of layers in `TransformerDecoder`.
    - **model.nheads** : Specify the number of attention heads.
- **`train_model.py`** : The main training code is written here. This uses the `train_config.yaml` file to read necessary details, reads preprocessed data from `data` directory and trains the model accordingly. This is invoked by `main.py` when given command line argument as `train`.

## How to put datasets
There are **two** data directories which contain the data.
### **raw_data**
As explained earlier, this contains the raw data i.e. raw images and the corresponding annotations for each dataset you want to use. The structure of this directory **must have to be** like this -
```
|
...
|
|--- raw_data
|      |
|      |--- Bongabdo
|      |      |
|      |      |--- Annotations
|      |      |--- Images
|      |
|      |--- BanglaWriting
|      |      |
|      |      |--- Annotations
|      |      |--- Images
|      |
|      |--- <Your dataset name>
|             |
|             |--- Annotations
|             |--- Images
```
Each dataset has a directory named after it and it contains handwritten script images inside *Images* directory and annotations/ground truths inside the *Annotations* directory. For simplicity, the images and corresponding ground truths are named as same. For example, image *abc.jpg* has its corresponding ground truth as *abc.txt*.
For the ease of use, I recommend you to download the datasets from the Kaggle links given below. These dataset repositories contain images and annotations in the above mentioned structure. Download the datasets and unzip inside the **`raw_data`** folder.
    
- [Bongabdo](https://www.kaggle.com/datasets/ayanwap7/bongabdo1429)
- [BanglaWriting](https://www.kaggle.com/datasets/ayanwap7/banglawriting-with-page-level-annotations)
- CMATERDb
- WBSUBNdb

### **data**
As explained earlier, this folder contains preprocessed data which are split into **Train** and **Validation** sets.
- **Train** folder contains training Images and Annotations in the respective folders.
- **Validation** folder contains validation Images and Annotations in the respecctive folders.

The split ratio can be defined in `preprocess.ipynb` file while creating the splits. The directory structure should look like this -
```
|
...
|
|--- data
|      |
|      |--- Train
|      |      |
|      |      |--- Annotations
|      |      |--- Images
|      |
|      |--- Validation
|             |
|             |--- Annotations
|             |--- Images
```

## How to Run the program

### 1. Preprocessing the data
1. Download the datasets from the links given above
2. Extract the datasets inside `raw_data` directory. The directory structure is given above.
3. Run the steps given in `preprocess.ipynb` notebook. Make necessary changes (such as split size), if applicable.
4. The preprocessed data will be populated inside `data` folder and split in `Train` and `Validation` sets.

### 2. Training a Model
1. Clone this repository into your local machine using **`git clone git@github.com:ayan-cs/bangla-ocr-transformer`**. You can also download the `.zip` file and extract it.
2. Make necessary changes in `train_config.yaml` file as per your choice/computing capacity.
3. Run `main.py` using the following command : **`python main.py train`**
4. The model will be trained. After successful training, trained model will be available inside `Saved Checkpoints` folder. Training logs, Error plot and Tokenizer file will be populated inside `Outputs` folder.

### 3. Inference
1. Identify the model you want to use for inference.
2. Make sure you have the **Validation** set available inside `data` folder.
3. Put the model name (without extension) inside `inference_config.yaml` file.
4. Run `main.py` using the following command : **`python main.py inference`**
5. After successful execution, Inference logs will be available inside `Outputs` directory.

### 4. Single Image Inference