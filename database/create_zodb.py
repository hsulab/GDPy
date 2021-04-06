#!/usr/bin/env python3
# -*- coding: utf-8 -*

import persistent

from GDPy.database.PersistentAtoms import Account


import ZODB, ZODB.FileStorage

storage = ZODB.FileStorage.FileStorage('mydata.fs')
db = ZODB.DB(storage)
connection = db.open()
root = connection.root

# Probably a bad idea:
#root.account1 = Account()

# add to application only
import BTrees
root.accounts = BTrees.OOBTree.BTree()
root.accounts['account-1'] = Account()

import transaction
transaction.commit()

db.close()
