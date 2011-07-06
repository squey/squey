CC=gcc
CFLAGS=-I../../../libpicviz/src/include/ `pkg-config apr-1 --cflags` -I./include/ -ggdb
LIBS=-L../../../libpicviz/src/ -lpicviz -lpcre `pkg-config apr-1 --libs` -ledit -ltermcap
DESTDIR=/usr/bin

all: textdig

textdig: textdig.c basic-foreach.c console.c functions-exec.c read-file.c
	$(CC) textdig.c basic-foreach.c console.c functions-exec.c read-file.c -o td $(CFLAGS) $(LIBS)

install:
	cp td $(DESTDIR)
