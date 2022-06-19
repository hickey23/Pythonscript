# from selenium import webdriver
# import time

# def dianzan(driver):
#     driver.find_element_by_class_name("点赞按钮的class name").click()
#     dianzan1 = driver.find_element_by_class_name("点赞按钮的class name").text # 查看下点击按钮上的数字
#     print(">>>:",dianzan1)

# n = 1
# while n <3:
#     options = webdriver.ChromeOptions()
#     options.add_experimental_option('excludeSwitches', ['enable-logging'])
#     driver = webdriver.Chrome(options=options)
#     driver.set_window_size(200, 200) # 设置浏览器窗口的大小
#     driver.delete_all_cookies() # 清除浏览器cookie
#     driver.get('URL') # 输入目标网页
#     dianzan(driver)
#     driver.delete_all_cookies() # 清除浏览器cookie
#     driver.quit() # 退出浏览器
#     time.sleep(5) # 休息5秒
#     n +=1
#创建一个鼠标
import time
from pynput import mouse,keyboard
time.sleep(5)
m_mouse=mouse.Controller()
m_keyboard = keyboard.Controller()
# m_mouse =mouse.controller
# (
m_mouse.click(mouse.Button.left)
while(True):
    for i in range(10000):
        m_keyboard.type('老子是你爹')
        m_keyboard.press (keyboard.Key.enter)
        m_keyboard.release(keyboard.Key.enter)
        time.sleep(0.3)
        if i==50:
            print('程序结束')
            break