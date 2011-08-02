/*
 * $Id: decode.cpp 1950 2011-02-20 14:58:34Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QString>
#include <QStringList>

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVNormalizer.h>

#include <iostream>

void display_help(void)
{

	std::cout << "Syntax: log2csv log-type filename\n";

}

int main(int argc, char **argv)
{
	QString logtype;
	QString logfile;

	// Normalize *normalize = new Normalize();


	if (argc <= 2) {
    		display_help();
    		return 1;
  	}

	logtype = QString(argv[1]);
	logfile = QString(argv[2]);

	// QList<QStringList> qt_nraw =  normalize->normalize(logtype, logfile);
	// normalize->normalized_debug(qt_nraw);

	// delete normalize;

	return 0;
}
