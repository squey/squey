/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterFloatFraction.h"


Inendi::PVMappingFilter::decimal_storage_type Inendi::float_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter* m)
{
	QString stmp(QString::fromUtf8(buf, size));
	return process_utf16((const uint16_t*) stmp.constData(), stmp.size(), m);
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::float_mapping::process_utf16(const uint16_t* buf, size_t size, PVMappingFilter* m)
{
	QString& str = static_cast<Inendi::PVMappingFilterFloatFraction*>(m)->th_qs().local();
	str.setRawData((QChar const*) buf, size);

	static QChar sep1('/');
	static QChar sep2('\\');

	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	int index_sep = str.indexOf(sep1);
	if (index_sep == -1) {
		index_sep = str.indexOf(sep2);
		if (index_sep == -1) {
			ret_ds.storage_as_float() = str.toFloat();
			return ret_ds;
		}
	}

	const float f = str.left(index_sep).toFloat();
	const float div = str.mid(index_sep+1).toFloat();
	float res;
	if (div == 0) {
		res = f;
	}
	else {
		res = f/div;
	}
	ret_ds.storage_as_float() = res;
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterFloatFraction)
