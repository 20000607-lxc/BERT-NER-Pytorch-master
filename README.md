## Chinese and English NER using GPT2 and Bert


### dataset list

1. cner: datasets/cner


2. CLUENER: datasets/cluener

**note**: donot have labels in the test dataset, to evaluate the test results, should subimit the results to the official link. the official link is  https://github.com/CLUEbenchmark/CLUENER

3. conll_03_english: datasets/conll_03_english

**note**: it's the BIO version of conll2003 dataset. get from https://github.com/Alibaba-NLP/CLNER 

4. ontonote: datasets/ontonote
   
**note**: the same with dataset used in [NAACL 2021] Better Feature Integration for Named Entity Recognition (In NAACL 2021) SynLSTM-for-NER/data/ontonotes at master · xuuuluuu/SynLSTM-for-NER 

5.ontonote4.0 

**note**: the dataset is for chinese ner and get from https://github.com/ShannonAI/glyce 


6. other datasets:

conll2003: datasets/conll2003

**note**: it's the BIESO version of conll2003 dataset, use it to test if this model is adaptable to other labeling style. 

BTC and GUM: I didn't test them yet.

**note**: please write another DataProcessor in `ner_seq.py`  to use the datasets except for 1,2,3
### model list

1. GPT2+Softmax

2. GPT2+CRF

### transformers

1.I use transformers 4.6.0  which is in `models.transformers_master `

2.The transformers used in the original project is still in `models.transformers` but it is the lower version and using it causes bugs.

3. for the chinese pretrained gpt2, I use the published model from "uer/gpt2-chinese-cluecorpussmall" 

### requirement

1. PyTorch == 1.7.0
2. cuda=9.0
3. python3.6+
4. transformers >= 4.6.0
5. use seqeval to compute the metric 

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
there are two prompt style:
1. (m,m,0) : construct the query as : prompt+input+prompt+input and use the output hidden state of the latter input to do classification 

2. (m,length_of_max_sequence_length,0) : construct the query as : prompt+input+prompt and use the output hidden state of the latter prompt to do classification
### run the code

1. Modify the configuration information in `run_ner_xxx.py`,  please only use `run_ner_softmax.py`
2. Modify the params in ` finetuning_argparse.py`
4. Modify the prompt template by setting `TEMPLATE_CLASSES` in `run_ner_xxx.py`.
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

**best results for gpt2**
1. cner:

evaluation:  acc: 0.94 - recall: 0.93 - f1: 0.93 
params: learning_rate=5e-5 weight_decay=0.01 template='1' model_type='chinese_pretrained_gpt2'

2. cluener:
   
evaluation: acc: 0.76 - recall: 0.74 - f1: 0.75
params: learning_rate=5e-5 weight_decay=0.01 template='1' model_type='chinese_pretrained_gpt2'


3. conll2003: 
   
evaluation:  acc: 0.94 - recall: 0.93 - f1: 0.93
params: learning_rate=5e-5 weight_decay=0.01 template='1' model_type='gpt2'

4. ontonote: 

evaluation:  acc: 0.85 - recall: 0.85 - f1: 0.85
params: learning_rate=1e-4 weight_decay=0.01 template='1' model_type='gpt2'

the other params are default values. # BERT-NER-Pytorch-master
# BERT-NER-Pytorch-master
