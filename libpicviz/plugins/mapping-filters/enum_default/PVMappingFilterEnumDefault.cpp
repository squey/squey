#ifdef WIN32
#include <float.h> // for _logb()
#endif
#include <math.h>

#include "PVMappingFilterEnumDefault.h"


size_t tbb_hasher(QString const& str)
{
	return qHash(str);
}

static float _enum_position_factorize(int enumber)
{
	float res = 0;
#ifdef WIN32
	int N = _logb(enumber);
#else
	int N = ilogb(enumber);
#endif
	int i;
	int x = enumber;

	if ( ! enumber) return -1;
	
	for (i = 0; i != N+1; i++) {
		if (x%2 == 0) {
			res = 2 * res;
		} else {
			res = 1+2*res;
		}
		x = x >> 1;
	}
	
	res = res / (float)pow((float)2, (int)N+1);
	
	return res;
}

void Picviz::PVMappingFilterEnumDefault::init_from_first(QString const& /*value*/)
{
	_enum_hash.clear();
	_poscount = 0;
}

float Picviz::PVMappingFilterEnumDefault::operator()(QString const& value)
{
	float retval = 0;
	int position = 0;

	hash_values::iterator it_v = _enum_hash.find(value);
	if (it_v != _enum_hash.end()) {
		position = (*it_v).second;
		retval = _enum_position_factorize(position);
	} else {
		int poscount = _poscount.fetch_and_increment() + 1;
		std::pair<hash_values::iterator,bool> ret = _enum_hash.insert(hash_values::value_type(value, poscount));
		if (!ret.second) { // That value has been inserted in the meanwhile by another thread
			poscount = ret.first->second;
		}
		retval = _enum_position_factorize(poscount);
	}

	return retval;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterEnumDefault)
