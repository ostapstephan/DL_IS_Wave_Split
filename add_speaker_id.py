# coding: utf-8

import pandas as pd

sm = pd.read_csv('meta.csv')
sm['id']= sm.groupby('speaker').ngroup()
sm.to_csv('meta_id.csv',index = False)
print('speaker labels added')
