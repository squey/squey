#include "PVMappingFilterStringSort.h"
#include <QVector>
#include <QByteArray>
#include <algorithm>
#include <vector>

typedef std::vector< std::pair<QByteArray,uint64_t> > vec_conv_sort_t;
typedef vec_conv_sort_t::value_type str_local_index;

static inline bool compLocal(const str_local_index& s1, const str_local_index& s2)
{
	return strcoll(s1.first.constData(), s2.first.constData()) < 0;
}

float* Picviz::PVMappingFilterStringSort::operator()(PVRush::PVNraw::nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	// This is so far the fastest way found to sort a vector of strings
	// Pre-converting in the current locale and using strcoll is really faster
	// than directly using QString::localeAwareCompare (under linux !)
	
	// Pre-conversion and save the original index
	vec_conv_sort_t v_local;
	v_local.reserve(values.size());
	for (size_t i = 0; i < values.size(); i++) {
		v_local.push_back(str_local_index(values[i].toLocal8Bit(),i));
	}

	std::sort(v_local.begin(), v_local.end(), compLocal);

	QByteArray prev;
	uint64_t cur_index = 0;
	uint64_t size = v_local.size();
	for (size_t i = 0; i < v_local.size(); i++) {
		str_local_index const& v = v_local[i];
		QByteArray const& str_local = v.first;
		uint64_t org_index = v.second;
		if (prev != str_local) {
			cur_index++;
			prev = str_local;
		}
		_dest[org_index] = (float)cur_index/(float)size;
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterStringSort)
