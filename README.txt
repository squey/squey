#
# @file
#

This is the source code for the INENDI Project. This project is built with various libraries. 
The goal of this README is to explain what each library is to understand the project better.

INENDI is designed to be flexible, modular and cross-plateform. While most projects like to
define themselves are modular, INENDI has real approach where most of the functions that can
be launched are plugins and each library has one big task, but one task only.

Since understanding the project means understanding this modularity, the README is the perfect
place to get started into the project.

libpvkernel: Core functions that are shared among all modules; Split any log file into an array

libinendi: Glues the low level elements together:
 * Handles the transformation of text information into positions
 * Apply selections
 * Handles layers
 * Handles various filters

inendi-inspector: QT GUI using all libraries above

helpers: various programs and libraries that can be useful in and around INENDI
