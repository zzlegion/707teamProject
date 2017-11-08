# run denoise
import subprocess

pixel_list = [96, 200]
kind_list = ['positive_', 'negative_']

for pixel in pixel_list:
    for kind in kind_list:
        posi_dir = toPath + kind + str(pixel)  
        nega_name_list = getImageName(nega_dir)
        de_posi_dir = posi_dir + '_de/'
        os.makedirs(de_posi_dir)
        for element in nega_name_list:
            subprocess.check_output(['python','denoise/src/main.py', '-i', posi_dir + '/' + element, 
                         '-o', de_posi_dir + element])
