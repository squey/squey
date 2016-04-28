/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVFormat.h>

#include <iostream>

void display_help(void)
{
	std::cout << "Syntax: pvformatcheck format-file\n";
}

int main(int argc, char** argv)
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
