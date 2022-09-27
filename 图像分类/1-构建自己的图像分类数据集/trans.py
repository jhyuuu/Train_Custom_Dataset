import os

path = os.getcwd()
print(path)

os.chdir(path)
retval = os.getcwd() 
dirlist = os.listdir(path)

print(dirlist)
for i in dirlist:
    if os.path.splitext(i)[1] == '.ipynb':
        cmd = 'jupyter nbconvert --to script' + ' ' + i
        os.system('%s' % (cmd))

