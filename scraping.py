import pandas as pd 
import requests 
from bs4 import BeautifulSoup 
import time 
# import re 
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 



base_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(base_dir, "cleaned_data.csv")

df = pd.read_csv(csv_file_path)
first_url = df.loc[0, "URLs"]

options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage") 
options.add_argument("start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def scrape_selenium(url): 
    driver.get(url) 
    time.sleep(2)
    
    driver.execute_script("window.scrollBy(0, 800);")
    time.sleep(1) 
    
    try: 
        see_more = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#bookDescription_feature_div a.a-expander-prompt"))
        )
        
        see_more.click()
        time.sleep(1)
        
        # try: 
        #     see_more = driver.find_element(By.CSS_SELECTOR, "a.a-expander-header.a-declarative")
        #     see_more.click()
        #     time.sleep(1) 
        # except: 
        #     pass
    except: 
        pass 
    
    
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # editorialReviews_feature_div
    # a-section a-spacing-small a-padding-small
    
    editorialReviews = soup.select_one("#editorialReviews_feature_div .a-section a-spacing-small a-padding-small")
    
    desc_div = soup.select_one("#bookDescription_feature_div")
    if desc_div: 
        return desc_div.get_text(" ", strip=True)
 
    partial_desc_div = soup.select_one(".a-expander-content.a-expander-partial-collapse-content")
    if partial_desc_div: 
        return partial_desc_div.get_text(" ", strip=True)
    
    product_facts = soup.select_one("#productFactsDesktop_feature_div .a-expander-content")
    if product_facts: 
        return product_facts.get_text(" ", strip=True)
    
    prod_desc = soup.select_one("#productDescription")
    if prod_desc: 
        return prod_desc.get_text(" ", strip=True)
    
    return "Description not found"


######## TEST FIRST URL #################
# description = scrape_selenium(first_url)

# print ("\n--- SCRAPED DESCRIPTION (SELENIUM) ---\n")
# print(description)

########### ITERATE FOR WHOLE TABLE ########### 
descriptions = []

for idx, row in df.iterrows():
    print(idx)
    url = row["URLs"]
    desc = scrape_selenium(url)
    
    # Remove "Read more" at end of description
    desc = desc.split()
    desc = desc[:-2]
    
    desc = " ".join(desc)
    
    descriptions.append(desc)
    
    time.sleep(random.uniform(1.5,3.5))
    print(desc)
    
df["Description"] = descriptions
df.to_csv("books_with_descriptions.csv", index=False)
    
    
    

# # for idx, row in df.iterrows():
#     # url = row["URLs"]
#     # desc = scrape(url)
#     # descriptions.append(desc)
    
#     # time.sleep(random.uniform(1.5, 3.5,))
    
# first_url = df.loc[0, "URLs"]
# print("Testing URL:", first_url)

# description = scrape(first_url)
# print("\n--- SCRAPED DESCRIPTION ---\n")
# print(description)

# # df["Description"] = descriptions

# # df.to_csv("books_with_descriptions.csv", index=False)
