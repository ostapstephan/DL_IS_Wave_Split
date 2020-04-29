# coding: utf-8
import pandas as pd

df = pd.read_csv('meta.csv')

all_mp3 = pd.DataFrame(df.loc_of_mp3.unique(),columns=['path'])
to_be_scraped = pd.read_csv('mp3_list.csv',names=['path'])

common = all_mp3.merge(scraped,on=['path'])

not_scraped = to_be_scraped[(~scraped.path.isin(common.path))]

not_scraped.reset_index(drop=True).to_csv('left2scrape.csv',index=False)
