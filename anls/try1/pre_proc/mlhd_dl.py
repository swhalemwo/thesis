from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
import time
import argparse
import os
import json

def setup(start, end):
    amt2 = "application/x-tar;application/x-gtar"
    profile = FirefoxProfile()
    profile.set_preference("browser.download.panel.shown", False)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", amt2)

    profile.set_preference("browser.helperApps.neverAsk.openFile","tar, archive, application/x-gzip, application/x-gtar, application/x-tgz, application/gzip, application/tar, application/tar+gzip")


    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.download.dir", '~/mlhd/dl/')

    from selenium.webdriver.firefox.options import Options
    options = Options()
    # options.headless = True
    
    driver = webdriver.Firefox(firefox_profile=profile, options=options)

    driver.get('http://bit.ly/MLHD-Dataset')

    print('wait till page loaded')
    time.sleep(5)
    

    # scroll down a bunch


    actions = ActionChains(driver)
    actions.send_keys(Keys.TAB).perform()

    while True:
        actions.send_keys(Keys.SPACE).perform()
        print('scrolling down a bit')
        time.sleep(2)

        links = get_links(driver, start, end)
        if len(links) == len(range(start, end)):
            break

    return(driver, links)

def get_links(driver, start, end):
    """get links"""
    
    aimed_secs = [i for i in range(start,end)]
    links = {}
    for a in driver.find_elements_by_xpath('.//a'):
        try:
            txt = a.get_attribute('href')
            sebgs = txt[103:106]
            if int(sebgs) in aimed_secs:
                links[sebgs] = a
                # links.append(a)
                print(a.get_attribute('href'))
        except:
            pass

    return(links)


# done_chnks = get_done_chunks()
# sebgs = "021"


def get_done_chunks():
    dones = os.listdir('/home/johannes/mlhd/us')
    return(dones)


def dl_fin_chck(sebgs):
    fl_nm = dl_dir + 'MLHD_' + sebgs + '.tar'
    fl_nm_part = dl_dir +'MLHD_' + sebgs + '.tar.part'
    
    dir_files = os.listdir(dl_dir)
    no_goods = 0

    fl_sz_old = 0
    
    while True: 
        try:
            print('getting filesize of part' + fl_nm_part)
            fl_sz = os.path.getsize(fl_nm_part)
            
            if fl_sz_old == fl_sz:
                no_goods+=1
            else:
                no_goods = 0
            print(no_goods)
                
            fl_sz_old = fl_sz

            if no_goods == 3:
                print('ABORT')
                res = 'fail'
                break
        
        except:

            try:
                fl_sz_fnl = os.path.getsize(fl_nm)
                print(fl_nm + ' final file here')
                res = 'success'
                break
            
            except:
                res = 'someting wong'
                break
                
        time.sleep(5)
        
    return(res)
    # print(sebgs + ' finished')
    
    
def fl_prep(sebgs):
    """ moves archive to specific dir, unpacks it and deletes base, calls other function to further process"""

    fl_nm = dl_dir + 'MLHD_' + sebgs + '.tar'
    # make own new directory
    dir_str = 'cd ' + gnr_dir + ' && mkdir ' + sebgs
    os.system(dir_str)

    mv_str = 'mv ' + fl_nm + " " + gnr_dir + sebgs + "/"
    os.system(mv_str)

    upk_str = 'cd ' + gnr_dir + sebgs + " && tar -xvf MLHD_" + sebgs + '.tar'
    os.system(upk_str)

    rm_bs_str = 'cd ' + gnr_dir + sebgs + " && rm MLHD_" + sebgs + '.tar'
    os.system(rm_bs_str)

    get_US_logs(gnr_dir + sebgs)
    
    
def get_US_logs(log_dir):

    """deletes all the non-us files, moves the remaining ones to US folder"""

    files1=os.listdir(log_dir)
    log_files = [i for i in files1 if i.endswith('.txt.gz')]
    log_files_ids = [i[0:36] for i in log_files]


    with open(gnr_dir + 'cri_dict.json', 'r') as fi:
        cri_dict = json.load(fi)

    # US_logs = []
    # nonUS_logs = []
    c = 0

    for i in log_files_ids:
        try:
            cri = cri_dict[i]
            if cri == 'US':

                pass
            else:
                del_str = 'cd ' + log_dir + ' && rm ' + i + '.txt.gz'
                os.system(del_str)
                c+=1
        except:
            del_str = 'cd ' + log_dir + ' && rm ' + i + '.txt.gz'
            os.system(del_str)
            # print(del_str)
            c+=1
            pass

    # mv_str2 = 'cd ' + log_dir + ' && mv *.txt.gz /home/johannes/mlhd/us/'
    mv_str2 = 'mv ' + log_dir + ' /home/johannes/mlhd/us/'
    
    os.system(mv_str2)

    # return(US_logs)
def cls_dl_win():
    parentWindow= driver.window_handles[0]
    all_windows = driver.window_handles
    non_main = [i for i in all_windows if i != parentWindow]
    for i in non_main:
        driver.switch_to_window(i)
        time.sleep(3)
        driver.close()
    driver.switch_to_window(parentWindow)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('start', help='start chunk id')
    parser.add_argument('end', help='end chunk id')

    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)

    dl_dir = '/home/johannes/mlhd/dl/'
    gnr_dir = '/home/johannes/mlhd/'
    
    driver, links = setup(start, end)
    print('setup done')

    # links = get_links(driver, start,end)

    for vlu in links.keys():
        done_chnks = get_done_chunks()
        if vlu in done_chnks:
            pass
        else:
            print(vlu)

            print(links[vlu])
            
            links[vlu].click()
            # close download window
            cls_dl_win()

            time.sleep(2)

            # ttl_no_goods = 0 
            while True:
                res = dl_fin_chck(vlu)
                if res == 'success':
                    break
                
                else:
                    clean_str = 'rm ' + dl_dir + '*'
                    os.system(clean_str)
                    driver.quit()                    
                    while True:
                        try:
                            driver, links2 = setup(start, end)
                            break
                        except:
                            print('go to sleep')
                            driver.quit()
                            time.sleep(40)
                            pass

                    links2[vlu].click()
                    # close download window
                    cls_dl_win()
                    

                    # ttl_no_goods+=1
                    # if ttl_no_goods ==3:
                    #     driver, links = setup(start, end)
                    
                    
            fl_prep(vlu)




# import gzip



# * scrap
# ** creation of usr country dict, no longer needed
# cri_dict_rows = client.execute('select uuid, country from usr_info')
# cri_dict = {}
# for i in cri_dict_rows:
#     cri_dict[i[0]] = i[1]
# with open(gnr_dir + 'cri_dict.json' , 'w') as fo:
#     json.dump(cri_dict, fo)
