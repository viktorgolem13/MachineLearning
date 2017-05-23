import sys, os
from PIL import Image

def main():

    directory = sys.argv[1]
    listOfImageNames = os.listdir(directory)
    print(listOfImageNames[0])


    for infile in listOfImageNames:
        f, e = infile.split('.')
        
        outfile = f + ".jpg"
        if infile != outfile:

            print(f)
            print(e)
            print(directory)
            try:
                Image.open(directory+'\\'+infile).save(directory+'\\'+outfile)
            except IOError:
                print ("cannot convert", infile)

main()