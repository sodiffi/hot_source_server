from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('data'):
    f.extend(filenames)
    break
print(f[0:10])