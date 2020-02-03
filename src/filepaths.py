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

    def __init__(self, depth=1, git=False):
        self.depth = depth
        self.refresh()

    def refresh(self):
        if self.depth:
            for level in range(self.depth):
                os.chdir('..')
        
        self.basepath = os.path.abspath('')

        for root, dirs, files in os.walk(self.basepath):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            print(root, dirs, files)
            print('\n')
            for folder in dirs:
                if folder not in dir(self):
                    setattr(self, folder, os.path.join(root, folder))

    def files(self, path):
        return [f for r, d, f in os.walk(path)][0]

    def dir_rerc(self):
        r, d, f = [_ for _ in os.walk(self.basepath)][0]
        if len(d):
            for folder in d:
                return self.dir_rerc()


if __name__ == "__main__":
    root = FilePaths(depth=1)
    print(root.dir_rerc())
    # print(root.notebooks)
    # print(dir(root))
    # for attr in dir(root):
        # print(attr, root.__getattribute__(attr))
    # print(root.clean.files())
