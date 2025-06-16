CC			=	gcc
INCLUDES    =	-I.

all : doublethree demomathslibrary.py  

# Build for the C only demonstrator for calling the library function 

doublethree : testmain.o libdemolibrary.so
	${CC} -o doublethree testmain.o -L. -ldemolibrary 

testmain.o : testmain.c libdemolibrary.so
	${CC} -c ${INCLUDES} testmain.c

demolibrarysource.o : demolibrarysource.c demolibrarysource.h
	${CC} -c ${INCLUDES} demolibrarysource.c

libdemolibrary.so : demolibrarysource.o
	${CC} -shared -o libdemolibrary.so demolibrarysource.o 

# Build for the python wrapper  

demomathslibrary.py : demolibrarysource.c demolibrarysource.h pythoninterface.i pythoninterface.py 
	swig -python -I. pythoninterface.i 
	python pythoninterface.py build_ext -i 

clean :
	rm -f *.o
	rm -f *.so
	rm -f doublethree
	rm -f pythoninterface_wrap.c
	rm -f demomathslibrary.py
	rm -rf build
	rm -rf __pycache__


        
