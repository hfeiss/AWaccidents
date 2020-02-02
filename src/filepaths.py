import os

# for root, dirs, file in os.walk(sourcepath):
#     list_years.extend(file)

# mappath = os.path.split(os.path.abspath(''))[0]
# srcpath = os.path.split(mappath)[0]
# rootpath = os.path.split(srcpath)[0]
# datapath = os.path.join(rootpath, 'data/')
# cleanpath = os.path.join(datapath, 'cleaned/')
# imagepath = os.path.join(rootpath, 'images/')

# os.listdir()

class FilePaths():

    def __init__(self, depth=0, git=False):
        self.depth = depth
        self.refresh()

    def refresh(self):
        if self.depth:
            for level in range(self.depth - 1):
                os.chdir('..')
        
        self.basepath = os.path.abspath('')

        # for folder in os.listdir(self.basepath):
        #     setattr(self, folder, os.path.join(self.basepath, folder))

        for root, dirs, files in os.walk(self.basepath):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            if dir not in dir(self):
                setattr(self, dir, os.path.join(self.basepath, root))

if __name__ == "__main__":
    paths = FilePaths(depth=0)
    print(paths.notebooks)
