import os
import cv2 as cv

# function to convert image from RGB to Grayscale
def convertRGBtoGray(src, dest):
    # read image and convert to grayscale
    image = cv.imread(src)
    gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    # write grayscale image to destination
    cv.imwrite(dest, gray_img)

cwd = os.getcwd()
# check if .zip containing images exists
if not os.path.isfile("./mirflickr25k.zip") and not os.path.isdir("./mirflickr"):
    os.system("wget http://press.liacs.nl/mirflickr/mirflickr25k.v3/mirflickr25k.zip")
    os.system("unzip mirflickr25k.zip")
elif not os.path.isdir("./mirflickr"):
    os.system("unzip mirflickr25k.zip")

os.chdir("./mirflickr")
dirs = ["train", "cross-validation", "test"]
subdirs = ["groundtruth", "grayscale"]

# check if subdirectories exist
for d in dirs:
    if (os.path.isdir("./train") and
       os.path.isdir("./test") and 
       os.path.isdir("./cross-validation")):
           print("Data already organized, exiting now")
           exit(0)

# create directories
for d in dirs:
    path = os.path.join(os.getcwd(), d)
    print(f'Creating {path}')
    os.mkdir(path)
    for s in subdirs:
        path = os.path.join(os.getcwd(), d, s)
        print(f'Creating {path}')
        os.mkdir(path)

if not os.path.isdir("./models"):
    path = os.path.join(os.getcwd(), "./models")
    os.mkdir(path)

cval_count = 0
test_count = 0
# organize images into train, test, and validation set
for i in range(0, 25000):
    src = f'im{i+1}.jpg'

    # 20,000 in training set
    # 2500 in cross validation set
    # 2500 in test set
    if i < 20000:
        dest = './train'
        count = i
    elif i < 22500:
        dest = './cross-validation'
        count = cval_count 
        cval_count += 1
    else:
        dest = './test'
        count = test_count 
        test_count += 1

    # move image to correct directory for that set
    print(f'creating {dest}/grayscale/gs{count}.jpg')
    convertRGBtoGray(src, f'{dest}/grayscale/gs{count}.jpg')
    print(f'creating {dest}/im{count}.jpg')
    os.system(f'mv {src} {dest}/groundtruth/gt{count}.jpg')
    #os.system(f'mv {src} {dest}/im{count}.jpg')

file_count = 0
for d in dirs:
    path = os.path.join(os.getcwd(), d)
    for s in subdirs:
        path = os.path.join(os.getcwd(), d, s)
        num_files = len([f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))])
        file_count += num_files
        print(f'Dir: {path} has {num_files} files')

print(f'Total: {file_count} files')
