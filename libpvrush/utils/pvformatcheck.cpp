/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <pvrush/PVFormat.h>

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

