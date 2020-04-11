# Bias in WAT

### Setup Environment
```shell script
conda create -n py37_venv python=3.7
conda activate py37_venv
pip install -r requirements.txt
```

### Data Preparation
Download data from [Google drive](https://drive.google.com/file/d/1MGO3xMdDUIVKrjOAsT7j8_l1EIOW6yCA/view?usp=sharing). 

Please note that data themselves are licensed under 
[Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License](
http://creativecommons.org/licenses/by-nc-nd/3.0/deed.en_US).
**They cannot be redistributed or used for commercial purposes.**
```shell script
mkdir data/
mv data-file-just-downloaded data/SWOW-EN.R100.csv.zip
unzip data/SWOW-EN.R100.csv.zip -d data/
```

### Preprocess
Pre-process the raw data.
```shell script
python preprocess.py
```

### Stereotype Propagation
Run stereotype propagation algorithm. 
You could specify the GPU device here.
Use CPU as default.
```shell script
python stereotype_propagation.py [--gpu 0]
```

### Lexicon
The stereotype lexicon is at ```data/stereotype_lexicon.txt```.
