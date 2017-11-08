# step1 : data cleaning
import os
import cv2

# extract all images
path = '/home/long/Desktop/DL/project/eyedata/'
toPath = '/home/long/Desktop/DL/project/data/'
directory = os.fsencode(path)
elements = []
for subdir, dirs, files in os.walk(directory):
    for file in files:
        elements.append(os.path.join(subdir, file))
        #os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")

def remove_prefix(s, prefix):
    return s[len(prefix):] if s.startswith(prefix) else s

def rename(element):
    element = remove_prefix(element, path)
    name_list = element.split('/')
    newname = name_list[0] + '_' + name_list[1] + '_' + name_list[-1]
    if name_list[2] == '0':
        return toPath + 'negative/' + newname
    elif name_list[2] == '2':
        return toPath + 'positive/' + newname

if __name__ == "__main__":
    # rename and orgnazed 
    for element in elements: 
        # transfer to string
        element = element.decode('UTF-8')
        if element.endswith('JPG'):
            # read image 
            image = cv2.imread(element)
            # rotate the image by 180 degrees if it is in right folder
            if 'right' in element:
                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, 180, 1.0)
                #rotated = cv2.warpAffine(image, M, (w, h))
                cv2.imwrite(element,cv2.warpAffine(image, M, (w, h)))
            else:
                cv2.imwrite(element, image)
            #print(rename(element))
            os.rename(element, rename(element))
