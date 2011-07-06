This is the source code for the Picviz Project. This project is built with various libraries. 
The goal of this README is to explain what each library is to understand the project better.

Picviz is designed to be flexible, modular and cross-plateform. While most projects like to
define themselves are modular, Picviz has real approach where most of the functions that can
be launched are plugins and each library has one big task, but one task only.

Since understanding the project means understanding this modularity, the README is the perfect
place to get started into the project.

libpvcore: Core functions that are shared among all modules

libpvrush: Split any log file into an array

libpvsample: Methods to investigate the array with limited samples, useful for big files

libpicviz: Glues the low level elements together:
 * Handles the transformation of text information into positions
 * Apply selections
 * Handles layers
 * Handles various filters

libpvgl: Manage OpenGL code, high performance graphical representation library

libpvwidgets: Provides widgets to deal with various datastructures of above libraries

picviz-inspector: QT GUI using all libraries above



