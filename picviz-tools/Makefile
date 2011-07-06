CC=gcc
CFLAGS=-I../libpicviz/src/include/ `pkg-config apr-1 --cflags` `python-config --cflags`
LIBS=-L../libpicviz/src/ -lpicviz -lpcre `pkg-config apr-1 --libs` `python-config --libs`

all: log2csv textdig

log2csv: log2csv.c
	$(CC) log2csv.c -o log2csv $(CFLAGS) $(LIBS)

#logmapped2csv: logmapped2csv.c
#	$(CC) logmapped2csv.c -o logmapped2csv $(CFLAGS) $(LIBS)

