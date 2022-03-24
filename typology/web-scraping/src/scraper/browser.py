# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:33:07 2020

@author: jpuig
"""
# Import the Selenium web driver
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

from  navigator import NavigatorScraper

class Browser:
    
    def __init__(self, path):          
      self.path = path
      self.browser = None
      self.navigator = None


    def start(self):
        ua = UserAgent()
        user_agent = ua.random
        print(user_agent)

        options = Options()
        options.add_argument(f'user-agent={user_agent}')                
        
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        self.browser.get(self.path)
        
        self.navigator = NavigatorScraper(self.browser)


    def get_current_page(self):
        return self.browser 

    def navigate(self):
        self.navigator.get_next_link()           

    def stop(self):
        self.browser.close()           