d={}
d[1]=0
try:
    d[0]
    print(True)
except KeyError:
    print(False)
    
print(d[1])