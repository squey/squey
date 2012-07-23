/**
 * \file pvformatcheck.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVFormat.h>

#include <iostream>

void display_help(void)
{

  std::cout << "Syntax: pvformatcheck format-file\n";

}

int main(int argc, char **argv)
{
	QString formatfile;
	PVRush::PVFormat format;

	if (argc <= 1) {
    		display_help();
    		return 1;
  	}

	formatfile = QString(argv[1]);
	format.populate_from_xml(formatfile);
	format.debug();

	return 0;
}

