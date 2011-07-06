CC=gcc
CFLAGS=-I../../libpicviz/src/include/ `pkg-config apr-1 --cflags` `python-config --cflags` `pkg-config cairo-png --cflags`
LIBS=-L../../libpicviz/src/ -lpicviz -lpcre `pkg-config apr-1 --libs` `python-config --libs` `pkg-config cairo-png --libs`
DESTDIR=/usr/bin

all: log2png

log2png: log2png.c
	$(CC) log2png.c -o log2png $(CFLAGS) $(LIBS)

install:
	cp log2png $(DESTDIR)
