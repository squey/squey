/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/file.h>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
#include "test-env.h"

	PVRush::File file("../../tests/files/pvkernel/rush/file_ending.utf16.gz");
	QByteArray qba;

	printf("code name=%s\n", file.codec->name().data());
	//   printf("is compressed=%d\n", file.is_compressed);

	file.Uncompress(QString("../../tests/files/pvkernel/rush/file_ending.utf16.gz"),
	                QString("outfile"));
	//   qba = file.file.read(12);
	//   printf("some data='%s'\n", qba.data());

	return 0;
}
