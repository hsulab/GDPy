#!/usr/bin/env python3
# -*- coding: utf-8 -*

import persistent

from GDPy.database.PersistentAtoms import Account


import ZODB, ZODB.FileStorage

storage = ZODB.FileStorage.FileStorage('mydata.fs')
db = ZODB.DB(storage)

with db.transaction() as conn:
    print(conn.root.accounts['account-1'].balance)

