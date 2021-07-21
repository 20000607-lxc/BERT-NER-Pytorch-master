## Chinese and English NER using GPT2 and Bert 


### dataset list

1. cner: datasets/cner
2. CLUENER: datasets/cluener

**note**: the official link is  https://github.com/CLUEbenchmark/CLUENERBut I didn't find the datset from the above link, instead, I download the cluener dataset from  https://github.com/liuyukid/transformers-ner
3. conll2003_bio: datasets/conll2003_bio

**note**: it's the BIO version of conll2003 dataset, but the `train.txt` file is too big, so I use `dev.txt` to do training and `test.txt` to do validation. 

4. other datasets:
   
conll2003: datasets/conll2003

**note**: it's the BIESO version of conll2003 dataset, use it to test if this model is adaptable to other labeling style. I got low acc with this dataset for now. 

BTC and GUM: I didn't test them yet. 

**note**: please write another DataProcessor in `ner_seq.py`  to use the datasets except for 1,2,3
### model list

1. GPT2+Softmax

2. GPT2+CRF

### transformers

1.I use  transformers 4.6.0  which is in `models.transformers_master ` 

2.The transfomers used in the original project is still in `models.transformers` but it is the lower version and using it causes many bugs. 


### requirement

1. PyTorch == 1.7.0
2. cuda=9.0
3. python3.6+
4. transformers >= 4.6.0

### input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.
The cner dataset labels are transferred into BIOS scheme in the DataProcessor. 
```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```
### promot format 
if template = (m,n,0), then the sequence fed into gpt2 is prompt1(length=m) + input + prompt2(length=n)


### run the code

1. Modify the configuration information in `run_ner_xxx.py` 
2. Modify the params in ` finetuning_argparse.py`
3. Notice that in ` finetuning_argparse.py`, I set --do_train = True,  --do-eval = False  --do_predict = False
which means only evaluate during training and do not predict.
4. Modify the prompt template by setting `TEMPLATE_CLASSES` in `run_ner_xxx.py` and you can choose any integers for x and m in template (x,m,0).
5. `BART_for_ner.py` cannot run for now.  

**note**: file structure of the model

```text
├── prev_trained_model
|  └── bert_base
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
|  |  └── ......
```
# gpt2_ner
