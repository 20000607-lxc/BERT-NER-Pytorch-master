""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids, tokens=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len
        self.tokens = tokens

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("andi611/bert-base-cased-ner")

def convert_examples_to_features(english, markup, tokenizer_name, task_name, examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    count = 0
    the_no_entity_number = 0
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    sum_length_of_example = 0
    if english:
        if 'gpt2' in tokenizer_name:
            print("gpt2_english tokenizer， use bios s for the lonely entity which contains one ****token***** ")
            for (ex_index, example) in enumerate(examples):
                if ex_index % 10000 == 0:
                    logger.info("Writing example %d of %d", ex_index, len(examples))
                if type(example.text_a) == list:
                    if example.text_a == []:# if list == []: pass!
                        continue
                    new_text = ' '.join(example.text_a)
                    tokens = tokenizer.tokenize(' ' + new_text)
                    sum_length_of_example += len(tokens)
                else:
                    raise(NotImplementedError)

                if len(tokens) == 0:# for the empty tokens list: pass!
                    count += 1# count such abnormal tokens
                    continue

                label_ids = [label_map[x] for x in example.labels]
                flag = 1
                for i in label_ids:
                    if i != 0:
                        flag = 0# 表示该example含有entity
                        break
                the_no_entity_number += flag

                # align the label_ids with tokens
                new_label = [0] * len(tokens)
                j = 0
                if 's' in markup:
                    # todo only for bios
                    for i in range(len(tokens)):
                        if 'Ġ' in tokens[i]:
                            new_label[i] = label_ids[j]
                            j = j+1
                        else:
                            if new_label[i-1] % 3 == 2:# B- label
                                new_label[i] = new_label[i-1]+1# new_label[i] should be I-
                            else:
                                new_label[i] = new_label[i-1]# new_label[i] should be I- or O
                                # should not use O(0 means "O") anymore!

                    # replace B- with S-
                    for i in range(len(new_label)-1):
                        # for all the lonely token(do not count the split words), replace B- with S-
                        if new_label[i] % 3 == 2 and new_label[i+1] == 0:# means new_label[i] == B- and new_label[i+1] == O
                            new_label[i] = new_label[i]-1# replace B- with S-

                    k = len(new_label)-1
                    if new_label[k] % 3 == 2:# means new_label[k] == B-, since it is the sentence from file, we assume its for the lonely token(there is nothing with it anymore)
                        new_label[k] = new_label[k]-1# replace B- with S-

                else:
                    for i in range(len(tokens)):
                        if 'Ġ' in tokens[i]:
                            new_label[i] = label_ids[j]
                            j = j+1
                        else:
                            if new_label[i-1] % 2 == 1:# B- label
                                new_label[i] = new_label[i-1]+1# new_label[i] should be I-
                            else:
                                new_label[i] = new_label[i-1]# new_label[i] should be I- or O
                                # should not use O(0 means "O") anymore!

                # truncate
                special_tokens_count = 0
                if len(tokens) > max_seq_length - special_tokens_count:
                    tokens = tokens[: (max_seq_length - special_tokens_count)]
                    new_label = new_label[: (max_seq_length - special_tokens_count)]
                segment_ids = [sequence_a_segment_id] * len(tokens)

                # # todo 1 仿照bert在input的前面后面加上特殊的fix-token（不随continuous prompt变化）目前看结果没什么变化 那就去掉吧
                # new_label += [label_map['O']]
                # segment_ids += [0]
                # if cls_token_at_end:
                #     new_label += [label_map['O']]
                #     segment_ids += [0]
                # else:
                #     new_label = [label_map['O']] + new_label
                #     segment_ids = [0] + segment_ids
                # gpt2 tokenizer 不添加cls和sep 且special_tokens_count=0

                pad_token = 0
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # input_ids += [102]
                # input_ids = [101]+input_ids

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                input_len = min(len(new_label), max_seq_length)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                    new_label = ([pad_token] * padding_length) + new_label
                else:
                    input_ids += [pad_token] * padding_length
                    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                    segment_ids += [pad_token_segment_id] * padding_length
                    new_label += [pad_token] * padding_length

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                # if ex_index < 5:
                #     logger.info("*** Example ***")
                #     logger.info("guid: %s", example.guid)
                #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                #     logger.info("label_ids: %s", " ".join([str(x) for x in new_label]))

                # if flag == 0:# todo 2 only use the sequence that contains entity
                features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                              segment_ids=segment_ids, label_ids=new_label))# tokens = tokens

            print("****************  the total no entity example number: "+str(the_no_entity_number)+'  ******************')
            print("****************  average length of examples(not truncated): "+str(sum_length_of_example/ex_index) + ' ******************')
            return features, count

        # elif "bert" or 'Bert' in tokenizer_name:
        #     print('bert english tokenizer')
        #     for (ex_index, example) in enumerate(examples):
        #         if ex_index % 10000 == 0:
        #             logger.info("Writing example %d of %d", ex_index, len(examples))
        #
        #         if type(example.text_a) == list:
        #             new_text = ' '.join(example.text_a)
        #             tokens = tokenizer.tokenize(new_text)
        #         label_ids = [label_map[x] for x in example.labels]
        #
        #         flag = 1
        #         for i in label_ids:
        #             if i != 0:
        #                 flag = 0
        #         the_no_entity_number += flag
        #
        #         # align the label_ids with tokens
        #         new_label = [0] * len(tokens)
        #         j = 0
        #         for i in range(len(tokens)):
        #             if '##' not in tokens[i]:
        #                 new_label[i] = label_ids[j]
        #                 j = j+1
        #                 if j == len(label_ids):
        #                     # ids that cannot be converted should be passed, such examples include:
        #                     # [' 's ', ...]
        #                     break# todo 这里到底那个地方出问题了？？？
        #             else:
        #                 new_label[i] = 0# new_label[i-1]
        #
        #         # Account for [CLS] and [SEP] with "- 2".
        #         special_tokens_count = 2
        #         if len(tokens) > max_seq_length - special_tokens_count:
        #             tokens = tokens[: (max_seq_length - special_tokens_count)]
        #             new_label = new_label[: (max_seq_length - special_tokens_count)]
        #
        #         tokens += [sep_token]
        #         new_label += [label_map['O']]
        #         segment_ids = [sequence_a_segment_id] * len(tokens)
        #
        #         if cls_token_at_end:
        #             tokens += [cls_token]
        #             new_label += [label_map['O']]
        #             segment_ids += [cls_token_segment_id]
        #         else:
        #             tokens = [cls_token] + tokens
        #             new_label = [label_map['O']] + new_label
        #             segment_ids = [cls_token_segment_id] + segment_ids
        #
        #         if len(tokens) > max_seq_length - special_tokens_count:
        #             tokens = tokens[: (max_seq_length - special_tokens_count)]
        #             new_label = new_label[: (max_seq_length - special_tokens_count)]
        #             segment_ids = segment_ids[: (max_seq_length - special_tokens_count)]
        #
        #         input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #         # The mask has 1 for real tokens and 0 for padding tokens. Only real
        #         # tokens are attended to.
        #         input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        #
        #         # Zero-pad up to the sequence length.
        #         padding_length = max_seq_length - len(input_ids)
        #         if pad_on_left:
        #             input_ids = ([pad_token] * padding_length) + input_ids
        #             input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        #             segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        #             new_label = ([pad_token] * padding_length) + new_label
        #         else:
        #             input_ids += [pad_token] * padding_length
        #             input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        #             segment_ids += [pad_token_segment_id] * padding_length
        #             new_label += [pad_token] * padding_length
        #
        #         assert len(input_ids) == max_seq_length
        #         assert len(input_mask) == max_seq_length
        #         assert len(segment_ids) == max_seq_length
        #
        #         # if ex_index < 5:
        #         #     logger.info("*** Example ***")
        #         #     logger.info("guid: %s", example.guid)
        #         #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #         #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #         #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #         #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #         #     logger.info("label_ids: %s", " ".join([str(x) for x in new_label]))
        #
        #         input_len = min(len(new_label), max_seq_length)
        #         features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
        #                                       segment_ids=segment_ids, label_ids=new_label))# tokens = tokens
        #
        #     print("the_no_entity_number: "+str(the_no_entity_number))
        #     return features, count
        else:
            raise(ValueError("tokenizer not implemented, English dataset only support gpt2 model and gpt2 tokenizer"))

    else:# 中文
        print("chinese:only use bert-base-chinese tokenizer")
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            if type(example.text_a) == list:
                new_text = ''.join(example.text_a)
            else:
                raise(NotImplementedError)

            tokens = tokenizer.tokenize(new_text)
            sum_length_of_example += len(tokens)
            label_ids = [label_map[x] for x in example.labels]

            flag = 1
            for i in label_ids:
                if i != 0:
                    flag = 0# 表示该example含有entity
                    break
            the_no_entity_number += flag

        # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            # todo test remove all special tokens (但是预训练的gpt2的vocabulary也是和bert一样的 可能没什么用）
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # label_ids += [label_map['O']]
            # label_ids = [label_map['O']] + label_ids
            # tokens = ['*']+tokens
            # tokens = tokens + ['*']
            # todo can add more fixed token to tell model to distiguash between input and prompt
            #  note: segment id is only used for computing loss

        # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens += [sep_token]
            label_ids += [label_map['O']]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [label_map['O']]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [label_map['O']] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            input_len = len(label_ids)
            # Zero-pad up to the sequence length.

            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert len(label_ids) == max_seq_length

            # if ex_index < 5:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s", example.guid)
            #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            if len(label_ids) == max_seq_length:
                features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                              segment_ids=segment_ids, label_ids=label_ids))# tokens = tokens
            else:
                count = count + 1
        print("****************   the total no entity example number: "+str(the_no_entity_number)+'  ******************')
        print("****************   average length of examples(not truncated): "+str(sum_length_of_example/ex_index) + '  ******************')
        return features, count

class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train", limit)

    def get_dev_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev", limit)

    def get_test_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test", limit)

    def get_labels(self):
        """See base class."""
        return ["O", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'X', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"]

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                # change the labels in cner dataset to BIO style
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train", limit)

    def get_dev_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev", limit)

    def get_test_examples(self, data_dir, limit):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "test", limit)
        # todo 文件中没有test.txt

    def get_labels(self):
        """See base class."""
        return ["O", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position','B-scene',"I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position', 'I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene', 'X', "[START]", "[END]"]

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class Conll2003Processor(DataProcessor):
    """Processor for an english ner data set."""

    def get_train_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train", limit)

    def get_dev_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev", limit)

    def get_test_examples(self, data_dir,limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test", limit)

    def get_labels(self, markup='bio'):
        """See base class.
       type can be choose from [bio bieso bios]"""
        if markup == 'bieso':
            raise(NotImplementedError)
            # return ['O',
            #         'S-LOC', 'B-LOC',  'I-LOC', 'E-LOC',
            #         'S-PER', 'B-PER',  'I-PER', 'E-PER',
            #         'S-MISC', 'B-MISC', 'I-MISC', 'E-MISC',
            #         'S-ORG', 'B-ORG', 'I-ORG', 'E-ORG'
            #         ] #'X', "[START]", "[END]"
            # note: should be in this order!
        elif markup == 'bios':
            return ['O',
                    'S-LOC', 'B-LOC',  'I-LOC',
                    'S-PER', 'B-PER',  'I-PER',
                    'S-MISC', 'B-MISC', 'I-MISC',
                     'S-ORG', 'B-ORG', 'I-ORG',
                    ] #'X', "[START]", "[END]"
        else:
            return ['O',
                    'B-LOC',  'I-LOC',
                    'B-PER',  'I-PER',
                    'B-MISC', 'I-MISC',
                    'B-ORG', 'I-ORG',
                    ] #'X', "[START]", "[END]"

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class OntonoteProcessor(DataProcessor):
    """Processor for an english ner data set."""

    def get_train_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.sd.conllx"), 'ontonote'), "train", limit)

    def get_dev_examples(self, data_dir, limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.sd.conllx"), 'ontonote'), "dev", limit)

    def get_test_examples(self, data_dir,limit=None):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.sd.conllx"), 'ontonote'), "test", limit)

    def get_labels(self, markup='bio'):
        """See base class.
        type can be choose from [bio bieso bios]"""
        if markup == 'bieso':
            raise(NotImplementedError)
            # return ["O",
            #         'S-NORP', 'B-NORP', 'I-NORP', 'E-NORP',
            #         'S-GPE', 'B-GPE', 'I-GPE', 'E-GPE',
            #         'S-FAC', 'B-FAC', 'I-FAC', 'E-FAC',
            #         'S-PERSON', 'B-PERSON',  'I-PERSON',  'E-PERSON',
            #         'S-DATE', 'B-DATE',  'I-PERSON', 'E-PERSON',
            #         'S-ORG', 'B-ORG', 'I-ORG', 'E-ORG',
            #         'S-LOC', 'B-LOC', 'I-LOC',  'E-LOC',
            #         'S-WORK_OF_ART', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'E-WORK_OF_ART',
            #         'S-EVENT', 'B-EVENT', 'I-EVENT', 'E-EVENT',
            #         'S-CARDINAL', 'B-CARDINAL', 'I-CARDINAL', 'E-CARDINAL',
            #         'S-ORDINAL', 'B-ORDINAL', 'I-ORDINAL', 'E-ORDINAL',
            #         'S-PRODUCT', 'B-PRODUCT', 'I-PRODUCT', 'E-PRODUCT',
            #         'S-QUANTITY', 'B-QUANTITY', 'I-QUANTITY', 'E-QUANTITY',
            #         'S-TIME', 'B-TIME', 'I-TIME', 'E-TIME',
            #         'S-PERCENT', 'B-PERCENT', 'I-PERCENT', 'E-PERCENT',
            #         'S-MONEY', 'B-MONEY', 'I-MONEY', 'E-MONEY',
            #         'S-LAW', 'B-LAW', 'I-LAW', 'E-LAW',
            #         'S-LANGUAGE', 'B-LANGUAGE', 'I-LANGUAGE',  'E-LANGUAGE',
            #         ] #'X', "[START]", "[END]"
        elif markup == 'bio':
            return ["O",
                    'B-NORP', 'I-NORP',
                    'B-GPE', 'I-GPE',
                    'B-FAC', 'I-FAC',
                    'B-PERSON',  'I-PERSON',
                    'B-DATE', 'I-DATE',
                    'B-ORG', 'I-ORG',
                    'B-LOC', 'I-LOC',
                    'B-WORK_OF_ART', 'I-WORK_OF_ART',
                    'B-CARDINAL', 'I-CARDINAL',
                    'B-ORDINAL', 'I-ORDINAL',
                    'B-PRODUCT', 'I-PRODUCT',
                    'B-QUANTITY', 'I-QUANTITY',
                    'B-TIME', 'I-TIME',
                    'B-EVENT', 'I-EVENT',
                    'B-PERCENT', 'I-PERCENT',
                    'B-MONEY', 'I-MONEY',
                    'B-LAW', 'I-LAW',
                    'B-LANGUAGE', 'I-LANGUAGE',
                    ]# 'X', "[START]", "[END]"
        else:
            return ["O",
                'S-NORP', 'B-NORP', 'I-NORP',
                'S-GPE', 'B-GPE', 'I-GPE',
                'S-FAC', 'B-FAC', 'I-FAC',
                'S-PERSON', 'B-PERSON',  'I-PERSON',
                'S-DATE', 'B-DATE', 'I-DATE',
                'S-ORG', 'B-ORG', 'I-ORG',
                'S-LOC', 'B-LOC', 'I-LOC',
                'S-WORK_OF_ART', 'B-WORK_OF_ART', 'I-WORK_OF_ART',
                'S-EVENT', 'B-EVENT', 'I-EVENT',
                'S-CARDINAL', 'B-CARDINAL', 'I-CARDINAL',
                'S-ORDINAL', 'B-ORDINAL', 'I-ORDINAL',
                'S-PRODUCT', 'B-PRODUCT', 'I-PRODUCT',
                'S-QUANTITY', 'B-QUANTITY', 'I-QUANTITY',
                'S-TIME', 'B-TIME', 'I-TIME',
                'S-PERCENT', 'B-PERCENT', 'I-PERCENT',
                'S-MONEY', 'B-MONEY', 'I-MONEY',
                'S-LAW', 'B-LAW', 'I-LAW',
                'S-LANGUAGE', 'B-LANGUAGE', 'I-LANGUAGE',
                ] #'X', "[START]", "[END]"
    # note: should be in this order!

    def _create_examples(self, lines, set_type, limit=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if limit != None:
                if i > limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

ner_processors = {
    "cner": CnerProcessor,
    'cluener': CluenerProcessor,
    'conll2003': Conll2003Processor,
    'ontonote': OntonoteProcessor

}
