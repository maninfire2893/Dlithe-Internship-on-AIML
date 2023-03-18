import requests
from bs4 import BeautifulSoup
url = "http://www.dlithe.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
about_us_link = soup.find("a", string="About us")
link_url=about_us_link.get('href')
link_res = requests.get(link_url)
link_soup=BeautifulSoup(link_res.text, 'html.parser')
s1=link_soup.find('p').get_text()
print(s1)
sentences = s1.split('.')[:5]
words = []
for sentence in sentences:
    words.extend(sentence.strip().split(' '))
print(words)
print(len(words))
print(words.count("DLithe"))
print(words.count("the")) 