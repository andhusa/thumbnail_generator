import os
import random
import shutil

#The given input folder should contain two folders of images. 
#The two folders are the two different classes and the classname should be given on its belonging folder.

#Example:

# dataset
#   |
#   |-classA
#   |    |-photo1.jpg
#   |    |-photo2.jpg
#   |    |-...
#   |
#   |-classB
#   |    |-photo1.jpg
#   |    |-photo2.jpg
#   |    |-...

def main():
    folder_path = "/global/D1/projects/soccer_clipping/closeUpSet"
    folder_path = "/global/D1/projects/soccer_clipping/AllsvenskanTestLogo"

    tempFolder = moveToTempFolder(folder_path)
    print(tempFolder)
    #create_dataset(tempFolder, "training", 0.8)  
    #create_dataset(tempFolder, "validation",0.75)
    create_dataset(tempFolder, "test", 1)

    outerFolder = tempFolder + "../"

    #sort_category(outerFolder, "training")
    #sort_category(outerFolder, "validation")
    sort_category(outerFolder, "test")

    os.rmdir(tempFolder)

def create_dataset(tempFolder, name, ratio, shuffle = True):        #function to split into train / test /validation
    
    data = os.listdir(tempFolder)
    if(shuffle):
        random.shuffle(data)
    print(tempFolder)
    print(name)
    newFolder = tempFolder + "../" + name
    os.mkdir(newFolder)
    file_to_be_moved = data[:int(len(data)*ratio)]
    for i in range(len(file_to_be_moved)):
        file = random.choice(os.listdir(tempFolder))
        temp_path = tempFolder + "/"+file
        shutil.move(temp_path,newFolder)

def sort_category(folder, name):                    #sort the images into different categories
    current_path = folder +'/'+name + "/"
    classes = []
    for f in os.listdir(current_path):
        className = f.split("_")[0]
        if className not in classes:
            classes.append(className)
    
    if(name == "test"):
        folder = current_path + "testClass"
        os.mkdir(folder)
        for i in os.listdir(current_path):
            name, ext = os.path.splitext(i)
            if ext == ".png" or ext == ".jpg":
                shutil.move(current_path + i,folder)
        return
    print(classes)
    for className in classes:
        os.mkdir(current_path + className)


    for i in os.listdir(current_path):
        name, ext = os.path.splitext(i)
        if ext == ".png" or ext == ".jpg":
            className = i.split("_")[0]
            shutil.move(current_path + i, current_path + className)
            
def moveToTempFolder(folder):
    if os.path.isdir(folder):
        if folder[-1] != "/":
            folder = folder + "/"
    else:
        print("The given destination is not a folder")
        return

    maxFolders = 2
    if len(os.listdir(folder)) == maxFolders:
        for f in os.listdir(folder):
            if not os.path.isdir(folder + f):
                print("not dir")
                print("The folder should only contain two folders")
                return
    else:
        print("length")
        print(len(os.listdir(folder)))
        print("The folder should only contain two folders")
        return
    
    
    newFolder = folder + "../" + folder.split("/")[-2] + "Structured/"
    tempFolder = newFolder + "temp/"
    if not os.path.exists(newFolder):
        os.mkdir(newFolder)
    else:
        raise Exception("Folder already exist, delete the existing folder if you want to create a new one")
    if not os.path.exists(tempFolder):
        os.mkdir(tempFolder)
    
    for folderName in os.listdir(folder):
        for img in os.listdir(folder + folderName):
            name, ext = os.path.splitext(img)
            if ext == ".png" or ext == ".jpg":
                folderWordLength = len(folderName) + 1
                if len(name) > folderWordLength:
                    if name[:folderWordLength] == folderName + "_":
                        shutil.copyfile(folder + folderName + "/" + img, tempFolder + img)
                    else:
                        shutil.copyfile(folder + folderName + "/" + img, tempFolder + folderName + "_"+ img)
                else:
                    shutil.copyfile(folder + folderName + "/" + img, tempFolder + folderName + "_"+ img)
            else:
                print("Unsupported file: " + ext)
                continue
    
    return tempFolder
                
    

if __name__ == "__main__":
    main()