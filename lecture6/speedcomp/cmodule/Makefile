CC=gcc
PYCFLAGS=`python3-config --cflags`
# NUMPYCFLAGS=-I/usr/lib/python3/dist-packages/numpy/core/include
CFLAGS=-Iinclude

.PHONY: all
all: bin/bessel.o bin/bessel.py.o lib/besselcc.so 


bin/bessel.o: src/bessel.c include/bessel.h
	$(CC) $(CFLAGS) src/bessel.c  -o $@ -O3 -lm -shared -fPIC

bin/bessel.py.o: src/bessel.py.c include/bessel.py.h
	$(CC) $(PYCFLAGS) $(CFLAGS) src/bessel.py.c -c -o $@ -fPIC

lib/besselcc.so: bin/bessel.py.o
	$(CC) $(PYCFLAGS) -fPIC -shared bin/bessel.py.o bin/bessel.o  -o $@ $(LDFLAGS) -lm

.PHONY: clean
clean: 
	rm bin/* lib/*


