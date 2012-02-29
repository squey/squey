#include "PVMappingFilterHostDefault.h"
#include <picviz/limits.h>
#include <pvkernel/core/PVStringUtils.h>
#include <pvkernel/core/network.h>

#include <QVector>
#include <QByteArray>

#include <algorithm>
#include <tbb/concurrent_vector.h>

#include <pvkernel/core/dumbnet.h>


typedef tbb::concurrent_vector< std::pair<QByteArray,uint64_t> > vec_conv_sort_t;
typedef vec_conv_sort_t::value_type str_local_index;
typedef tbb::concurrent_vector<uint32_t> list_indexes;

static inline bool compLocal(const str_local_index& s1, const str_local_index& s2)
{
	return strcoll(s1.first.constData(), s2.first.constData()) < 0;
}

Picviz::PVMappingFilterHostDefault::PVMappingFilterHostDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter(),
	_case_sensitive(false) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterHostDefault, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterHostDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-domain-lowercase", "Convert domain strings to lower case")].setValue<bool>(true);
	return args;
}

void Picviz::PVMappingFilterHostDefault::set_args(PVCore::PVArgumentList const& args)
{
	Picviz::PVMappingFilter::set_args(args);
	_case_sensitive = !args["convert-lowercase"].toBool();
}

float* Picviz::PVMappingFilterHostDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	int64_t ssize = values.size();
	//vec_conv_sort_t v_local;
	//v_local.reserve(ssize);
	float max_str = STRING_MAX_YVAL;
	list_indexes str_idxes;
	str_idxes.reserve(ssize);
	for (int64_t i = 0; i < ssize; i++) {
		QString v = values[i].get_qstr();
		uint32_t ipv4_v;
		if (PVCore::Network::ipv4_aton(v, ipv4_v)) {
			// IPv4 are mapped from 0 to 0.5
			_dest[i] = (float) (((double)ipv4_v/(double)(PICVIZ_IPV4_MAXVAL))/((double)2.0));
		}
		else {
			float res = PVCore::PVStringUtils::compute_str_factor(values[i].get_qstr(), _case_sensitive); 
			if (res > max_str) {
				max_str = res;
			}
			_dest[i] = res;
			str_idxes.push_back(i);
			//v_local.push_back(str_local_index(v.toLocal8Bit(),i));
		}
	}

	list_indexes::const_iterator it;
	max_str *= 2.0;
	for (it = str_idxes.begin(); it != str_idxes.end(); it++) {
		_dest[*it] = _dest[*it]/max_str + 0.5;
	}

#if 0 // Too slow for now, need improvements
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
#endif

	return _dest;
}

IMPL_FILTER(Picviz::PVMappingFilterHostDefault)
