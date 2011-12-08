#include "PVMappingFilterEnumDefault.h"

#ifdef WIN32
#include <float.h> // for _logb()
#endif
#include <math.h>


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

float* Picviz::PVMappingFilterEnumDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	float retval = 0;
	int position = 0;
	hash_values enum_hash;
	if (_grp_value && _grp_value->isValid()) {
		PVLOG_DEBUG("(mapping-enum) using previous values for enumeration\n");
		enum_hash = _grp_value->value<hash_values>();
	}
	_poscount = 0;

	for (size_t i = 0; i < values.size(); i++) {
		PVCore::PVUnicodeString const& value(values[i]);
		hash_values::iterator it_v = enum_hash.find(value);
		if (it_v != enum_hash.end()) {
			position = it_v.value().toInt();
			retval = _enum_position_factorize(position);
		} else {
			_poscount++;
			enum_hash[value] = QVariant((qlonglong)_poscount);
			retval = _enum_position_factorize(_poscount);
		}
		_dest[i] = retval;
	}

	if (_grp_value) {
		_grp_value->setValue<hash_values>(enum_hash);
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterEnumDefault)
