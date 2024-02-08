CC=gcc
PYCFLAGS=`python3-config --cflags` -fPIC
# NUMPYCFLAGS=-I/usr/lib/python3/dist-packages/numpy/core/include
LDFLAGS=-lm
CFLAGS=-Iinclude

.PHONY: all
all: bin/bessel.o bin/bessel.py.o lib/besselcc.so 


bin/bessel.o: src/bessel.c include/bessel.h
	$(CC) $(CFLAGS) src/bessel.c -c  -o $@ -O3

bin/bessel.py.o: src/bessel.py.c include/bessel.py.h
	$(CC) $(PYCFLAGS) $(CFLAGS) src/bessel.py.c -c -o $@

lib/besselcc.so: bin/bessel.py.o
	$(CC) $(PYCFLAGS) -shared bin/bessel.py.o bin/bessel.o  -o $@ $(LDFLAGS)

.PHONY: clean
clean: 
	rm bin/* lib/*


