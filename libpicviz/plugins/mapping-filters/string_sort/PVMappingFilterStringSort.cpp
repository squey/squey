#include "PVMappingFilterStringSort.h"
#include <algorithm>
#include <vector>
#include <tbb/parallel_sort.h>

typedef std::vector< std::pair<QByteArray,uint64_t> > vec_conv_sort_t;
typedef vec_conv_sort_t::value_type str_local_index;

typedef std::vector< std::pair<PVCore::PVUnicodeString const*, uint64_t> > vec_field_sort_t;
typedef vec_field_sort_t::value_type field_local_index;

static inline bool compLocal(const str_local_index& s1, const str_local_index& s2)
{
	return strcoll(s1.first.constData(), s2.first.constData()) < 0;
}

static inline bool compField(const field_local_index& s1, const field_local_index& s2)
{
	return s1.first->compare(*(s2.first)) < 0;
}

Picviz::PVMappingFilterStringSort::PVMappingFilterStringSort(PVCore::PVArgumentList const& args):
	PVMappingFilter()
{
	INIT_FILTER(PVMappingFilterStringSort, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterStringSort)
{
	PVCore::PVArgumentList args;
	return args;
}

float* Picviz::PVMappingFilterStringSort::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	// Make a memory-based comparaison and do not take care of the locale !
	vec_field_sort_t vec;
	vec.resize(values.size());
	for (uint64_t i = 0; i < values.size(); i++) {
		PVCore::PVUnicodeString const* f = &values[i];
		vec[i].first = f;
		vec[i].second = i;
	}

	tbb::parallel_sort(vec.begin(), vec.end(), compField);

	/*
	// This is so far the fastest way found to sort a vector of strings
	// Pre-converting in the current locale and using strcoll is really faster
	// than directly using QString::localeAwareCompare (under linux !)
	
	// Pre-conversion and save the original index
	vec_conv_sort_t v_local;
	v_local.reserve(values.size());
	for (size_t i = 0; i < values.size(); i++) {
		v_local.push_back(str_local_index(values[i].get_qstr().toLocal8Bit(),i));
	}

	std::sort(v_local.begin(), v_local.end(), compLocal);*/

	//QByteArray prev;
	PVCore::PVUnicodeString const* prev = NULL;
	uint64_t cur_index = 0;
	uint64_t size = vec.size();
	for (size_t i = 0; i < size; i++) {
		field_local_index const& v = vec[i];
		PVCore::PVUnicodeString const* str = v.first;
		uint64_t org_index = v.second;
		if (prev == NULL || prev->size() != str->size() || *prev != *str) {
			cur_index++;
			prev = str;
		}
		_dest[org_index] = (float)cur_index/(float)size;
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterStringSort)
