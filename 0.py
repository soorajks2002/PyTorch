file = ".py"

for i in range(3,10) :
    fil = str(i)+file
    with open(fil, "x") as f:
        f.write(None)