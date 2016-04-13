/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/inendi_assert.h>

int main()
{
	std::string refstr("salut");
	PVCore::PVUnicodeString unistr((const PVCore::PVUnicodeString::utf_char*) refstr.data(), refstr.size());

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
				int ret = PVCore::PVUnicodeString(a, strlen(a)).compare(PVCore::PVUnicodeString(b, strlen(b)));
				PV_VALID(ret, ress[i], "a", a, "b", b);
				++i;
				return ret < 0;
			});

	for (size_t i = 0; i < size; ++i) {
		PV_VALID(strs[i], sorted_strs[i]);
	}

	return 0;
}
