# BanglaOCR : Full Page Bangla handwritten text recognition using Image to Sequence Architecture

### Contents
Have a look at the brief explanations of what this repository folder/files contain and what their significance are.

- **charset** : Bangla, being an Indic language, I have used `utf-8` encoding for the representation purpose. The system uses **Character-level vocabulary**, and therefore, it is important to give our system a list of what the possible characters it is going to see. Thus, this directory contains two files :
    - `printable1.txt` : This file contains all possible (list NOT exhaustive) valid characters that can appear while writing Bangla.
    - `printable2.txt` : This file contains all possible (list NOT exhaustive) valid punctuations that can be used while writing any language including Bangla.
- **data** : This directory contains the data that we are going to use for Training the model as well as for the Validation purpose.

### How to put datasets


### How to Run the program