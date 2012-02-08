#include <common/common.h>
#include <code_bz/types.h>
#include <code_bz/init.h>

#include <cstdlib>

void init_random_bcodes(PVBCode* ret, size_t n)
{
	PVBCode tmp;
	for (size_t i = 0; i < n; i++) {
		tmp.int_v = rand();
		tmp.s.__free = 0;
		tmp.s.type %= 6;
		ret[i] = tmp;
	}
}

void init_constant_bcodes(PVBCode* ret, size_t n)
{
	PVBCode tmp;
	tmp.int_v = rand();
	tmp.s.__free = 0;
	tmp.s.type %= 6;
	for (size_t i = 0; i < n; i++) {
		ret[i] = tmp;
	}
}
