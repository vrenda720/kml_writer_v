import os

class dir_crawler:
    def __init__(self, target_dir, ext, outfile):
        self.target_dir = target_dir    # Root Directory, traverse starts here
        self.ext = ext                  # Look for this kind of file, or "all" for all files
        self.outfile = outfile          # Writeout files to this location. Pass None to return files rather than write them to file
        self.filepaths = []

    def traverse(self):
        stash_cwd = os.getcwd()
        queue = []
        queue.append(os.path.abspath(self.target_dir))
        for item in queue:
            os.chdir(item)
            for name in os.listdir(item):
                if os.path.isdir(name):
                    queue.append(os.path.abspath(name))
                elif os.path.isfile(name):
                    text,ext = os.path.splitext(name)
                    if ext == self.ext or self.ext == "all":
                        self.filepaths.append(os.path.abspath(name))
        self.filepaths.sort()
        os.chdir(stash_cwd)
        try:
            with open(self.outfile, 'w') as f:
                for file_path in self.filepaths:
                    f.write("{}\n".format(file_path))
        except TypeError:
            return self.filepaths
    
    def get_dirs(self):
        dir_list = []
        for file in self.filepaths:
            this_dir = os.path.dirname(file)
            if this_dir not in dir_list:
                dir_list.append(this_dir)
        return dir_list
    
