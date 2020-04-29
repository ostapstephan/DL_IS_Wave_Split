import eyed3
import os
import random
import sys
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_books(browser, size_book, next_page):
    btns = browser.find_elements_by_class_name('download-btn')
    try:
        num_mb= [float(b.text.replace('(','').replace(')','')[9:][:-2]) for b in btns]

        if len(btns) == len(num_mb):
            for i in range(len(btns)):
                if num_mb[i]<50:
                    # print(btns[i])
                    print(btns[i].find_element_by_xpath('a').get_attribute('href') )
                    # btns[i].click()
            browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")

            sleep(random.uniform(1, 10))

        else:
            print('error: mismatched length')

    except Exception as e:
        print('error page '+str(next_page)+':'+str(e))

    # go to the next page
    try:
        raise Exception('test','mytest')
        browser.find_element_by_link_text(str(next_page)).click() 

    except:
        browser.get(f'https://librivox.org/search?primary_key=1&search_category=language&search_page={str(next_page) }&search_form=get_results')

    sleep (random.gauss(5,3))

    return browser

def find_books():
    '''
    this is basically main for extracting the audiobooks that are less than 50 mb 
    '''

    desired_caps = {'prefs': {'download': {'default_directory': '/home/ostap/Documents/DL_Ind_Study/audiobooks/meta'}}}
    
    browser = webdriver.Chrome(desired_capabilities = desired_caps)
    browser.get('https://librivox.org/search?primary_key=1&search_category=language&search_page=1&search_form=get_results')
    
    sleep(10)
    for i in range(2 ,1000):
        size_book = 50
        browser = get_books(browser,size_book,i )


def myfind(in_str,to_find): 
     ret = in_str.find(to_find) 
     if ret == -1: 
         return 30000 
     else: 
         return ret 



def parse_speaker(text):
    text = text.lower()
    st = text.find('read ') 

    end = min(myfind(text[st:],'\n'),myfind(text[st:],';'),250)
    reader = text[st:st+end]

    st1 = myfind(reader,'by ')
    speaker = reader[st1+3:]

    return speaker

def write_out(x,f):
    f.write(','.join(map(str,x))+'\n') 

def get_metadata(browser):
    title_and_author_xpath = '//*[@id="maincontent"]/div[4]/div/div/div[2]'
    reader_xpath = '//*[@id="descript"]'

    try:
        author = browser.find_element_by_xpath('//*[@id="maincontent"]/div[4]/div/div/div[2]/dl/dd/span').text
        title = browser.find_element_by_xpath('/html/body/div[1]/main/div[4]/div/div/div[2]/h1/span').text
        text = browser.find_element_by_xpath('/html/body/div[1]/main/div[5]/div/div/div[1]/div[4]').text
        speaker = parse_speaker(text )

    except Exception as e:
        print('error '+ str(browser.current_url)  +' '+ str(e))
    
    # sleep (5+random.gauss(5,3))
    [author,title,speaker] = [author.replace(',','_' ).replace(' ','_' ), \
            title.replace(',','_' ).replace(' ','_' ), \
            speaker.replace(',','_' ).replace(' ','_' ) ] 
    return [speaker,author,title] 

def grab_book_metadata(link):
    '''
    Take link and return the title, author, and speaker
    '''

    chrome_options = Options()
    #chrome_options.add_argument("--no-sandbox") # linux only
    chrome_options.add_argument("--headless")
    browser = webdriver.Chrome(options=chrome_options)
    browser.get(link)
    
    # print(driver.page_source.encode("utf-8"))
    out = get_metadata(browser)

    # from IPython import embed
    # embed()
     
    browser.quit()

    return  out


# if __name__ == "__main__": 
    # # find_books()
    # base_path= '/share/audiobooks/mp3/'
    
    # files = os.listdir(base_path)
    # print(len(files))
    # for chapter in files:
        # loc = os.path.join(base_path,chapter)
        # audiofile = eyed3.load(loc)

        # link = audiofile.tag.comments[0].text
         
        # metadata = grab_book_metadata(link)
        # write_out([loc,link]+metadata)

        



