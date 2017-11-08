# step2-1: getImageName
def getImageName(directory_string):
    elements = []
    result = []
    directory = os.fsencode(directory_string)
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            elements.append(os.path.join(subdir, file))
    for element in elements: 
        # transfer to string
        element = element.decode('UTF-8')
        if element.endswith('JPG'):
            result.append(remove_prefix(element, directory_string))
            
    return result

import PIL
from PIL import Image

def resize(input_name, out_name, num_pixel):
    img = Image.open(input_name)
    img = img.resize((num_pixel, num_pixel))
    img.save(out_name)

# create a set of image with different resolution
pixel = 64
pixel_list = [32, 64, 96, 200]
for pixel in pixel_list:
    # resize image in negative folder
    for element in nega_name_list:
        resize('data/negative/' + element, 'data/negative_' + str(pixel) + '/' + element, pixel)
    # resize image in positive folder
    for element in posi_name_list:
        resize('data/positive/' + element, 'data/positive_' + str(pixel) + '/' + element, pixel)


if __name__ == "__main__":
    # extract all images
    nega_dir = toPath + 'negative/'
    posi_dir = toPath + 'positive/'  
    nega_name_list = getImageName(nega_dir)
    posi_name_list = getImageName(posi_dir)
