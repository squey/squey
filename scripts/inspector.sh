#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

export LD_LIBRARY_PATH=$(/usr/bin/realpath $(dirname $0)):$LD_LIBRARY_PATH
export PATH=$(/usr/bin/realpath $(dirname $0)):$PATH

# Migration from picviz to inendi
if [ ! -d $HOME/.inendi ] && [ -d $HOME/.picviz ];
then
    mv $HOME/.picviz $HOME/.inendi
    ln -s $HOME/.inendi $HOME/.picviz
fi

if [ ! -d $HOME/.config/ESI\ Group/ ] && [ -d $HOME/.config/Picviz\ Labs/ ];
then
        mv $HOME/.config/Picviz\ Labs/ $HOME/.config/ESI\ Group
	mv $HOME/.config/ESI\ Group/Picviz\ Inspector.conf $HOME/.config/ESI\ Group/INENDI\ Inspector.conf
	ln -s $HOME/.config/ESI\ Group $HOME/.config/Picviz\ Labs/
fi


inendi-inspector
