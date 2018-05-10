import sys
import os
from shutil import copyfile

#new_string = '# Gamma= 90.0 ; T= 0.3536 ; dt= 0.01\n'
new_string = '# Gamma= -- ; T= -- ; dt= 0.01\n'
replace_string = '# ================'

# Make a list of folders we want to process
folder_list = []
for i in range(len(sys.argv)):
    if os.path.isdir(sys.argv[i]):
        folder_list.append(sys.argv[i])
    elif i != 0:
        print(sys.argv[i] + ' is not a folder')

# Make a list of subfolders X0.00... in each folder
subfolder_lists = []
for folder in folder_list:
    sf_list = []
    for item in os.walk(folder):
        # subfolder name and contained files
        sf_list.append((item[0], item[2]))
    sf_list = sf_list[1:]
    subfolder_lists.append(sf_list)

for ifold, folder in enumerate(folder_list):
    print('==========================================================')
    print(folder)
    print('==========================================================')
    for isf, sf in enumerate(subfolder_lists[ifold]):
        sf_words = sf[0].split('/')
        print(sf_words[-1])
        folder_path = sf[0]
        copyfile(folder_path + '/log.txt', folder_path + '/log_orig.txt')
        f_orig = open(folder_path + '/log_orig.txt', 'r')
        f = open(folder_path + '/log.txt', 'w')
        for line in f_orig.readlines():
            pattern_len = len(replace_string)
            if line[:pattern_len] == replace_string:
                f.write(new_string)
            else:
                f.write(line)
        f_orig.close()
        f.close()

