

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
import time



# driver.get("https://www.google.com")

def setup(aimed_secs):
    amt2 = "application/x-tar;application/x-gtar"
    profile = FirefoxProfile()
    profile.set_preference("browser.download.panel.shown", False)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", amt2)

    profile.set_preference("browser.helperApps.neverAsk.openFile","tar, archive, application/x-gzip, application/x-gtar, application/x-tgz, application/gzip, application/tar, application/tar+gzip")


    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.download.dir", '~/mlhd/dl/')


    driver = webdriver.Firefox(firefox_profile=profile)

    driver.get('http://bit.ly/MLHD-Dataset')

    # scroll down a bunch
    actions = ActionChains(driver)
    for _ in range(8):
        actions.send_keys(Keys.SPACE).perform()
        time.sleep(1)

    # get links
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


aimed_secs = [i for i in range(20,30)]


done_links = []
done_chnks = get_done_chunks()

for l in list(links.keys()):
    print(l)
    

    # l.click()

dl_dir = '/home/johannes/mlhd/dl/'
gnr_dir = '/home/johannes/mlhd/'

# sebgs = "021"


for vlu in links.keys():
    if vlu in done_chnks:
        pass
    else:
        print(vlu)
    
        links[vlu].click()

        time.sleep(5)
        dl_fin_chck(sebgs)
        fl_prep(sebgs)


def get_done_chunks():
    dones = os.listdir('/home/johannes/mlhd/us')
    return(dones)


def dl_fin_chck(sebgs):
    fl_nm = dl_dir + 'MLHD_' + sebgs + '.tar'
    fl_nm_part = dl_dir +'MLHD_' + sebgs + '.tar.part'
    
    dir_files = os.listdir(dl_dir)

    while True: 
        try:
            fl_sz = os.path.getsize(fl_nm_part)
            
        except:
            print('part file not there')
            try:
                fl_sz_fnl = os.path.getsize(fl_nm)
                print('final file here')
                break
            
            except:
                pass
                
        time.sleep(5)
        
    print(sebgs + ' finished')
    
    
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

    # creation no longer needed
    # cri_dict_rows = client.execute('select uuid, country from usr_info')
    # cri_dict = {}
    # for i in cri_dict_rows:
    #     cri_dict[i[0]] = i[1]
    # with open(gnr_dir + 'cri_dict.json' , 'w') as fo:
    #     json.dump(cri_dict, fo)

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





# testfile = '/home/johannes/mlhd/11/2b1a9c83-231b-40cd-be09-a11ea381fd3b.txt.gz'

# slines = []
# with gzip.open(testfile,'rt') as f:
#     for line in f:
#         slines.append(line)
        




import gzip
