#include "PVMappingFilterHostDefault.h"
#include <picviz/limits.h>

#include <QVector>
#include <QByteArray>

#include <algorithm>
#include <tbb/concurrent_vector.h>

#include <pvcore/network.h>

#include <dnet.h>


typedef tbb::concurrent_vector< std::pair<QByteArray,uint64_t> > vec_conv_sort_t;
typedef vec_conv_sort_t::value_type str_local_index;

static inline bool compLocal(const str_local_index& s1, const str_local_index& s2)
{
	return strcoll(s1.first.constData(), s2.first.constData()) < 0;
}

float* Picviz::PVMappingFilterHostDefault::operator()(PVRush::PVNraw::nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	int64_t ssize = values.size();
	vec_conv_sort_t v_local;
	v_local.reserve(ssize);
#pragma omp parallel for
	for (int64_t i = 0; i < ssize; i++) {
		QString const& v = values[i];
		uint32_t ipv4_v;
		if (PVCore::Network::ipv4_aton(v, ipv4_v)) {
			// IPv4 are mapped from 0 to 0.5
			_dest[i] = (float) (((double)ipv4_v/(double)(PICVIZ_IPV4_MAXVAL))/((double)2.0));
		}
		else {
			v_local.push_back(str_local_index(v.toLocal8Bit(),i));
		}
	}

	// Sort the strings that represents hosts
	std::sort(v_local.begin(), v_local.end(), compLocal);

	// And map them from 0 to 0.5
	QByteArray prev;
	uint64_t cur_index = 0;
	size_t size = v_local.size();
	for (size_t i = 0; i < size; i++) {
		str_local_index const& v = v_local[i];
		QByteArray const& str_local = v.first;
		uint64_t org_index = v.second;
		if (prev != str_local) {
			cur_index++;
			prev = str_local;
		}
		_dest[org_index] = (float) cur_index;
	}
	float div = 2*(cur_index);
	for (size_t i = 0; i < size; i++) {
		uint64_t org_index = v_local[i].second;
		_dest[org_index] = _dest[org_index]/div + 0.5;
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterHostDefault)
