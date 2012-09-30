/**
 * \file pvstring.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString16.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <iostream>

int main()
{
	QString refstr("salut");
	PVCore::PVUnicodeString16 unistr((const PVCore::PVUnicodeString16::utf_char*) refstr.constData(), refstr.size());
	std::cout << unistr.compare("salut") << " " << refstr.compare("salut") << std::endl;
	std::cout << unistr.compare("salu") << " " << refstr.compare("salu") << std::endl;
	std::cout << unistr.compare("salute") << " " << refstr.compare("salute") << std::endl;

	const char* strs[] = {"Alarme","éa","Éb","FTW","test","Coralie","Damned","1924test"};
	const size_t size = sizeof(strs)/sizeof(const char*);
	std::cout << size << std::endl;
	std::sort(strs, strs+size, [=](const char* a, const char* b)
			{
				std::cout << "Compare " << a << " and " << b << ": ";
				int ret = PVCore::PVUnicodeString(a, strlen(a)).compareNoCase(PVCore::PVUnicodeString(b, strlen(b)));
				std::cout << ret << std::endl;
				return ret < 0;
			});
	for (const char* s: strs) {
		std::cout << s << std::endl;
	}

	return 0;
}
