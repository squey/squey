/**
 * \file pvstring.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString16.h>
#include <pvkernel/core/PVUnicodeString.h>
#include <iostream>

#include <pvkernel/core/picviz_assert.h>

int main()
{
	QString refstr("salut");
	PVCore::PVUnicodeString16 unistr((const PVCore::PVUnicodeString16::utf_char*) refstr.constData(), refstr.size());

	PV_VALID(unistr.compare("salut"), 0);
	PV_VALID(refstr.compare("salut"), 0);

	PV_VALID(unistr.compare("salu"), 1);
	PV_VALID(refstr.compare("salu"), 1);

	PV_VALID(unistr.compare("salute"), -1);
	PV_VALID(refstr.compare("salute"), -1);

	/* sorted_strs must be sorted by hand (sorry:)
	 * ress must be initialized according to the sorting algorithm, hoping it won't change :-)
	 */
	const char* strs[] = {"Alarme","éa","Éb","FTW","test","Coralie","Damned","1924test"};
	const char* sorted_strs[] = {strs[7], strs[0], strs[5], strs[6], strs[1], strs[2], strs[3], strs[4]};
	const int ress[]   = {1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1 };

	const size_t size = sizeof(strs)/sizeof(strs[0]);

	size_t i = 0;
	std::sort(strs, strs+size, [=, &i, &ress](const char* a, const char* b)
			{
				int ret = PVCore::PVUnicodeString(a, strlen(a)).compareNoCase(PVCore::PVUnicodeString(b, strlen(b)));
				PV_VALID(ret, ress[i], "a", a, "b", b);
				++i;
				return ret < 0;
			});

	for (size_t i = 0; i < size; ++i) {
		PV_VALID(strs[i], sorted_strs[i]);
	}

	return 0;
}
