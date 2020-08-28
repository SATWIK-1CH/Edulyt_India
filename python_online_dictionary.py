#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import requests


# ## BeautifulSoup is a Python package for working with real-world and broken HTML
#

# In[ ]:


from bs4 import BeautifulSoup as bs


# # load the website's source code read the word from the input arguments and append to the url

# In[ ]:


url = "https://www.dictionary.com/browse/"
try:
    word = input()
    url += word
except:
    print("mention a word")
    exit(-1)


#
# # Malking sure the internet connection is on

# In[ ]:



try:
    r = requests.get(url)
    soup = bs(r.content, 'lxml')#lxml is used as parser employed by BeautifulSoup
except:
    print("Check your Internet and try again!")
    exit(-1)


# # parse the source to obtain all necessary info

# In[ ]:



try:
    pos = soup.findAll("span", {"class": "luna-pos"})[0].text #Pos represents parts of speech
    answer_list = soup.findAll("ol")[0] #ol stands for ordered lists
    meanings = answer_list.findChildren("li", recursive=False) #used for finding all direct children and recursive is #used as children of children should not be considered
except:
    print("Word not found!")
    exit(-1)


# In[ ]:


#display the results
print(word + ":" + pos)

for (i, meaning) in enumerate(meanings):
    print()
    print(str(i + 1) + ".", meaning.text)




print("code by satwik cheppala")



