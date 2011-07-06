CC=gcc
CFLAGS=-I../src/include/ `pkg-config apr-1 --cflags` `python-config --cflags`
LIBS=-L../src/ -lpicviz `pkg-config apr-1 --libs` `python-config --libs`

all: area simple input view datatree2view

area: area.c
	$(CC) area.c -o area $(CFLAGS) $(LIBS)

simple: simple.c
	$(CC) simple.c -o simple $(CFLAGS) $(LIBS)

input: input.c
	$(CC) input.c -o input $(CFLAGS) $(LIBS)

view: view.c
	$(CC) view.c -o view $(CFLAGS) $(LIBS)

datatree2view: datatree2view.c
	$(CC) datatree2view.c -o datatree2view $(CFLAGS) $(LIBS)
