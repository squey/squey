#!/usr/bin/python

import sys

from cpicviz.datatree import *

#from pycviz import Source, Mapped, Plotted
import cairo

if len(sys.argv) < 4:
    sys.exit("Usage: %s logtype logfile out.png" % sys.argv[0])

logtype = sys.argv[1]
logfile = sys.argv[2]
pngfile = sys.argv[3]

print "Starting CPicViz construction"
datatree = DataTree("log2png")
scene = datatree.new_item("Scene", datatree)
source = datatree.new_item("Source", scene, logtype, logfile)
mapped = datatree.new_item("Mapped", source)
plotted = datatree.new_item("Plotted", mapped)
print "CPicViz part finished"

PADDING = 20
WIDTH = (400 * (source.get_column_count() - 1)) + (2 * PADDING)
HEIGHT = 1000

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)
ctx.rectangle(0,0,WIDTH,HEIGHT)
ctx.set_source_rgba(1, 1, 1, 1)
ctx.fill()
ctx.set_source_rgb(0, 0, 0)
ctx.set_line_width(0.5)

#Draw axes
i = 0
while i < source.get_column_count():
    ctx.move_to((i * 400) + PADDING, 0)
    ctx.line_to((i * 400) + PADDING, HEIGHT)
    ctx.stroke()

    i += 1

print "Number of rows:%d" % source.get_row_count()

# Draw linse
line = 0
while line < source.get_row_count():
    colpos = 0
    if (logtype == "pcap"):
        if source.get_value(line, 4) == "tcp":
            ctx.set_source_rgb(1, 0, 0)
        if source.get_value(line, 4) == "udp":
            ctx.set_source_rgb(0, 1, 0)

    while colpos < (source.get_column_count() - 1):
        x = (colpos * 400) + PADDING
        y = (1 - plotted.get_value(line, colpos)) * HEIGHT
        ctx.move_to(x, y)
        # ctx.show_text(str(plotted.get_value(line, colpos)))
        x = ((colpos + 1) * 400) + PADDING
        y = (1 - plotted.get_value(line, colpos + 1)) * HEIGHT
        ctx.line_to(x, y)
        # if ((colpos + 1) == (source.get_column_count() - 1)):
        #     ctx.show_text(str(plotted.get_value(line, colpos)))

        ctx.stroke()
        
        colpos += 1



    ctx.set_source_rgb(0, 0, 0)        

    line += 1

surface.write_to_png (pngfile)

