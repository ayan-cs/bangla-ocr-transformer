# BanglaOCR : Full Page Bangla handwritten text recognition using Image to Sequence Architecture

### Contents
Have a look at the brief explanations of what this repository folder/files contain and what their significance are.

- **charset** : Bangla, being an Indic language, I have used `utf-8` encoding for the representation purpose. The system uses **Character-level vocabulary**, and therefore, it is important to give our system a list of what the possible characters it is going to see. Thus, this directory contains two files :
    - `printable1.txt` : This file contains all possible (list NOT exhaustive) valid characters that can appear while writing Bangla.
    - `printable2.txt` : This file contains all possible (list NOT exhaustive) valid punctuations that can be used while writing any language including Bangla.
- **data** : This directory contains the data that we are going to use for Training the model as well as for the Validation purpose. For the directory structure, check out the next section.
- **Outputs** : This directory will contain *Tokenizer* file, *Training logs*, *Validation logs* and *Error plots* generated during training. You do not need to touch/modify this directory.
- **raw_data** : This will contain your raw data, which will be preprocessed before training/inference. You have to keep your dataset inside this directory. Data inside this directory can be preprocessed using `preprocess.ipynb` file. Throughout the process, raw images and ground truths will remain intact. As mentioned earlier, preprocessed images and ground truths will be stored inside **`data`** directory and from there, they will be used for training and inference.
- **Saved Checkpoints** : This directory will contain the trained models. The naming convention followed for naming the trained model is - `BanglaOCR_<No. of Epochs>_<Hidden layer dimension>_<No. of heads>_<No. of decoder layers>.pth`. For example, if a model name is `BanglaoCR_200_256_4_4.pth`, that means the model is trained for *200 epochs*, the model's *hidden layer dimension is 256*, it used *4 attention heads* and the number of *Transformer Decoder layer(s) is 4*.

During Inference, you need to copy the model name

### How to put datasets


### How to Run the program