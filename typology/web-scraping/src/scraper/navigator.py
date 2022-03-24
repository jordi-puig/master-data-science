# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:03:08 2020

@author: jpuig
"""
class NavigatorScraper:
  def __init__(self, browser):          
      self.browser = browser
      
  def get_next_link(self):
      class_name = 'tr-next-link'

      try:
          next_link = self.browser.find_element_by_class_name(class_name)
          next_link.click()
      except:
          raise Exception('not exists the next link')
          

  def get_previous_link(self):
      class_name = 'tr-prev-link'
      previous_link = self.browser.find_element_by_class_name(class_name)
      previous_link.click()  