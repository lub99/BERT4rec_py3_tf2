from __future__ import print_function
from collections import defaultdict
import re


def data_partition(train_dataset_name, test_dataset_name):
    session_num = 0
    itemnum = 0
    user_train = defaultdict(list)
    user_test = defaultdict(list)
    # assume user/item index starting from 1
    with  open(train_dataset_name, 'r') as f:
        for line in f:
            sid, i = re.split("\s+", line.rstrip())
            sid = int(sid)
            i = int(i)
            session_num = max(sid, session_num)
            itemnum = max(i, itemnum)

            user_train[sid].append(i)

    with open(test_dataset_name, "r") as test:
        for line in test:
            sid, i = re.split("\s+", line.rstrip())
            sid = int(sid)
            i = int(i)

            user_test[sid].append(i)
    return [user_train, user_test, session_num, itemnum]
