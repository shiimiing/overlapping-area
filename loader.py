import os
path = "C:/Users/ADMIN/Desktop/document/junx/Private_Test/videos"
dir_list = os.listdir(path)
for item in dir_list:
    cam_dir = path+"/"+item
    