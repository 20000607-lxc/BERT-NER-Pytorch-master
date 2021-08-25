# def markup_for_bert_chinese(markup, tokens, new_label):
# 没有必要转化 一律按照benchmark的做法采用biso
# 这里的代码没写错
#     if markup == 'biso':
#         # replace B- with S-
#         for i in range(len(new_label)-1):
#             # for all the lonely token(do not count the split words), replace B- with S-
#             if new_label[i] % 3 == 2 and new_label[i+1] == 0:# means new_label[i] == B- and new_label[i+1] == O
#                 new_label[i] = new_label[i]-1# replace B- with S-
#
#         k = len(new_label)-1
#         if new_label[k] % 3 == 2:# means new_label[k] == B-, since it is the sentence from file, we assume its for the lonely token(there is nothing with it anymore)
#             new_label[k] = new_label[k]-1# replace B- with S-
#
#     elif markup == 'bio':
#         return tokens, new_label
#
#     elif markup == 'bieso':
#         # replace B- with S- and I- with E-
#         for i in range(len(new_label)-1):
#             # for all the lonely token(do not count the split words), replace B- with S-
#             if new_label[i] % 4 == 2 and new_label[i+1] == 0:# means new_label[i] == B- and new_label[i+1] == O
#                 new_label[i] = new_label[i]-1# replace B- with S-
#             if new_label[i] % 4 == 3 and new_label[i+1] == 0:# means new_label[i] == I- and new_label[i+1] == O
#                 new_label[i] = new_label[i]+1# replace I- with E-
#
#         k = len(new_label)-1
#         if new_label[k] % 4 == 2:# means new_label[k] == B-, since it is the sentence from file, we assume its for the lonely token(there is nothing with it anymore)
#             new_label[k] = new_label[k]-1# replace B- with S-
#         if new_label[k] % 4 == 3:# means new_label[k] == I-
#             new_label[k] = new_label[k]+1# replace I- with E-
#
#     return tokens, new_label


def markup_for_gpt2_english(markup, tokens,  label_ids, label_all_tokens):
    assert markup in ['bio', 'bieso']
    j = 0
    new_label = [0] * len(tokens)
    # if markup == 'biso':
    #     for i in range(len(tokens)):
    #         if 'Ġ' in tokens[i]:
    #             new_label[i] = label_ids[j]
    #             j = j+1
    #         else:
    #             if new_label[i-1] % 3 == 2:# B- label
    #                 new_label[i] = new_label[i-1]+1# new_label[i] should be I-
    #             else:
    #                 new_label[i] = new_label[i-1]# new_label[i] should be I- or O
    #                 # should not use O(0 means "O") anymore!
    #
    #     # replace B- with S-
    #     for i in range(len(new_label)-1):
    #         # for all the lonely token(do not count the split words), replace B- with S-
    #         if new_label[i] % 3 == 2 and new_label[i+1] == 0:# means new_label[i] == B- and new_label[i+1] == O
    #             new_label[i] = new_label[i]-1# replace B- with S-
    #
    #     k = len(new_label)-1
    #     if new_label[k] % 3 == 2:# means new_label[k] == B-, since it is the sentence from file, we assume its for the lonely token(there is nothing with it anymore)
    #         new_label[k] = new_label[k]-1# replace B- with S-

    if markup == 'bio':
        for i in range(len(tokens)):
            if 'Ġ' in tokens[i]:
                new_label[i] = label_ids[j]
                j = j+1
            else:
                if label_all_tokens:
                    new_label[i] = new_label[i-1]
                    # 下面这种实验跑出来conll88% 不知道是不是他的问题  transformers 没有改变类型
                    # if new_label[i-1] % 2 == 1:# B- label
                    #     new_label[i] = new_label[i-1]+1# new_label[i] should be I-
                    # else:
                    #     new_label[i] = new_label[i-1]# new_label[i] should be I- or O
                    #     # should not use O(0 means "O") anymore!
                else:
                    new_label[i] = -100# todo note： the convention is -100 not O!!

    elif markup == 'bieso':
        label_ids = iob_iobes(label_ids)
        # # replace B- with S- and I- with E-
        #
        # for i in range(len(new_label)-1):
        #     # for all the lonely token(do not count the split words), replace B- with S-
        #     if new_label[i] % 4 == 2 and new_label[i+1] == 0:# means new_label[i] == B- and new_label[i+1] == O
        #         new_label[i] = new_label[i]-1# replace B- with S-
        #     if new_label[i] % 4 == 3 and new_label[i+1] == 0:# means new_label[i] == I- and new_label[i+1] == O
        #         new_label[i] = new_label[i]+1# replace I- with E-
        #
        # k = len(new_label)-1
        # if new_label[k] % 4 == 2:# means new_label[k] == B-, since it is the sentence from file, we assume its for the lonely token(there is nothing with it anymore)
        #     new_label[k] = new_label[k]-1# replace B- with S-
        # if new_label[k] % 4 == 3:# means new_label[k] == I-
        #     new_label[k] = new_label[k]+1# replace I- with E-

        for i in range(len(tokens)):
            if 'Ġ' in tokens[i]:
                new_label[i] = label_ids[j]
                j = j+1
            else:
                if label_all_tokens:
                    new_label[i] = new_label[i-1]
                    # 下面这种实验跑出来conll88% 不知道是不是他的问题 transformers 没有改变类型
                    # #这里可以索引i-1因为第一个单词必须是有G的
                    # if new_label[i-1] % 4 == 2:# B- label
                    #     new_label[i] = new_label[i-1]+1# todo ? new_label[i] should be I-
                    # elif new_label[i] % 4 == 3:
                    #     if i == len(new_label)-1:
                    #         new_label[i] = new_label[i-1]+1
                    #     elif new_label[i+1] == 0:# new_label[i] should be E-
                    #         new_label[i] = new_label[i-1]+1
                    # else:
                    #     new_label[i] = new_label[i-1]# new_label[i] should be I- or O
                    #     # should not use O(0 means "O") anymore
                else:
                    new_label[i] = -100

    return markup, tokens, new_label, label_ids
