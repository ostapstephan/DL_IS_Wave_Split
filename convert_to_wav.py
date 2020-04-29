import eyed3
import librosa
import soundfile as sf
import sys

import pandas as pd
import os 
import time
from pydub import AudioSegment
import scrape_audio

# convert mp3 to wav8k but split up 
def convert_to_8k(src,sample_len,meta,f):
    sr = 8000 
    y, s = librosa.load(src, sr=sr )
    dst = src.replace('/mp3/','/wav8k_split/')
    
    # cuts 25 sec 
    if len(y)>25*8000:
        clip = y[25*8000:]
    else: 
        return -1

    slices = []
    for i in range(int(clip.shape[0]/(sr*sample_len))):
        slices.append(clip[i*sr*sample_len:(i+1)*sr*sample_len])

    for i in range(int(clip.shape[0]/(sr*sample_len))):
        # append to the meta file 
        fn_s = dst.replace('.mp3', f'_{i}.wav')
        sf.write(fn_s , slices[i], 8000, 'PCM_16')
        scrape_audio.write_out(meta+[fn_s],f)

    return (dst)

# load file
# cut 25 sec off the start 
# break up file
# save to csv reader and other meta data and book name and fn 
# save file as wav8k  with pcm_16

# audiofile = eyed3.load(f"{sys.argv[1]}")
# link = audiofile.tag.comments[0].text
# author = audiofile.tag.artist
# title = audiofile.tag.title
# [audiofile,link, author, title]

if __name__ == "__main__":
    # base_path= '/share/audiobooks/mp3/'
    # base_path = sys.argv[1]
    # files = os.listdir(base_path)
    # print(len(files))
    sample_len = 5
    ''' ignore this cuz you accidentally stopped the scraper
    with open('./error.csv', 'a') as fn_error: 
        with open('./meta.csv', 'a') as f: 
            f.write('speaker,author_int,title_int,loc_of_mp3,link,author_mp3,title_mp3,loc_of_wav8k\n')


            for chapter in files:
                try:
                    print(chapter)
                    loc = os.path.join(base_path,chapter)
                    audiofile = eyed3.load(loc)
                    link = audiofile.tag.comments[0].text
                    author_mp3 = audiofile.tag.artist.replace(',','_').replace(' ','_')
                    title_mp3 = audiofile.tag.title.replace(',','_').replace(' ','_')

                    metadata = scrape_audio.grab_book_metadata(link)
                    metadata+=[loc,link,author_mp3,title_mp3]

                    convert_to_8k(loc,sample_len,metadata,f)
                except Exception as e:
                    fn_error.write(f'{chapter},{e}')
    '''            

    with open('./error.csv', 'a') as fn_error: 
        df = pd.read_csv('remaining.csv')
        with open('./meta.csv', 'a') as f: 
            for i in df.path.iteritems():
                chapter=i[1]
                print(i[1])

                # f.write('speaker,author_int,title_int,loc_of_mp3,link,author_mp3,title_mp3,loc_of_wav8k\n')
                # for chapter in files:
                try:
                    # loc = os.path.join(base_path,chapter)
                    loc = i[1]

                    audiofile = eyed3.load(loc)
                    link = audiofile.tag.comments[0].text
                    author_mp3 = audiofile.tag.artist.replace(',','_').replace(' ','_')
                    title_mp3 = audiofile.tag.title.replace(',','_').replace(' ','_')

                    metadata = scrape_audio.grab_book_metadata(link)
                    metadata+=[loc,link,author_mp3,title_mp3]

                    convert_to_8k(loc,sample_len,metadata,f)

                except Exception as e:
                    fn_error.write(f'{i[1]},{e}\n')
                
         
