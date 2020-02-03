import os
from addict import Dict


def rerc(path, ignore_hidden=True):
    result = Dict()
    root = [x for x in os.walk(path)][0][0]
    dirs = [x for x in os.walk(path)][0][1]
    all_files = [x for x in os.walk(path)][0][2]
    
    if ignore_hidden:
        files = [f for f in all_files if f[0] != '.' and f[0] != '_']
        dirs[:] = [d for d in dirs if d[0] != '.' and d[0] != '_']
    else:
        files = all_files

    result['files'] = files
    result['path'] = root
    result['dirs'] = dirs
    for folder in dirs:
        result[folder] = rerc(os.path.join(root, folder))
    
    return result

class FilePaths():

    def __init__(self, depth=1, ignore_hidden=True):
        self.depth = depth
        self.ignore_hidden = ignore_hidden
        self.set_basepath()

    def set_basepath(self):
        if self.depth:
            for _ in range(self.depth):
                os.chdir('..')
        
        self.basepath = os.path.abspath('')

    def return_dict(self):
        return rerc(self.basepath, self.ignore_hidden)


if __name__ == "__main__":
    root = FilePaths(depth=1).return_dict()
    print(root['files'])

