/**
 * \file pvstring.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString.h>
#include <iostream>

int main()
{
	QString refstr("salut");
	PVCore::PVUnicodeString unistr((const PVCore::PVUnicodeString::utf_char*) refstr.constData(), refstr.size());
	std::cout << unistr.compare("salut") << " " << refstr.compare("salut") << std::endl;
	std::cout << unistr.compare("salu") << " " << refstr.compare("salu") << std::endl;
	std::cout << unistr.compare("salute") << " " << refstr.compare("salute") << std::endl;

	return 0;
}
