/**
 * \file PVPlottingFilter.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVPlottingFilter.h>
#include <pvkernel/core/stdint.h>

Picviz::PVPlottingFilter::PVPlottingFilter() :
	PVFilter::PVFilterFunctionBase<uint32_t*, mapped_decimal_storage_type const*>(),
	PVCore::PVRegistrableClass<Picviz::PVPlottingFilter>(),
	_decimal_type(PVCore::FloatType)
{
	_dest = NULL;
	_dest_size = 0;
	_mandatory_params = NULL;
}

uint32_t* Picviz::PVPlottingFilter::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	int64_t size = _dest_size;
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = operator()(values[i]);
	}

	return _dest;
}

uint32_t Picviz::PVPlottingFilter::operator()(mapped_decimal_storage_type const /*value*/)
{
	PVLOG_WARN("In default plotting filter: returns 0 !\n");
	return 0;
}

void Picviz::PVPlottingFilter::set_mapping_mode(QString const& mode)
{
	_mapping_mode = mode;
}

void Picviz::PVPlottingFilter::set_dest_array(PVRow size, uint32_t* array)
{
	assert(array);
	_dest_size = size;
	_dest = array;
}

void Picviz::PVPlottingFilter::set_mandatory_params(Picviz::mandatory_param_map const& params)
{
	_mandatory_params = &params;
}

QStringList Picviz::PVPlottingFilter::list_modes(QString const& type, bool only_expandable)
{
	LIB_CLASS(PVPlottingFilter)::list_classes const& pl_filters = LIB_CLASS(PVPlottingFilter)::get().get_list();
	LIB_CLASS(PVPlottingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = pl_filters.begin(); it != pl_filters.end(); it++) {
		if (only_expandable && !it.value()->can_expand()) {
			continue;
		}
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << params[1];
		}
	}
    return ret;
}

QList<Picviz::PVPlottingFilter::p_type> Picviz::PVPlottingFilter::list_modes_lib(QString const& type, bool only_expandable)
{
	LIB_CLASS(PVPlottingFilter)::list_classes const& pl_filters = LIB_CLASS(PVPlottingFilter)::get().get_list();
	LIB_CLASS(PVPlottingFilter)::list_classes::const_iterator it;
	QList<p_type> ret;
	ret.reserve(pl_filters.size());
	for (it = pl_filters.begin(); it != pl_filters.end(); it++) {
		if (only_expandable && !it.value()->can_expand()) {
			continue;
		}
		QString const& name = it.key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << it.value();
		}
	}
    return ret;
}

QString Picviz::PVPlottingFilter::mode_from_registered_name(QString const& rn)
{
	QStringList params = rn.split('_');
	return params[1];
}

QString Picviz::PVPlottingFilter::get_human_name() const
{
	return mode_from_registered_name(registered_name());
}

void Picviz::PVPlottingFilter::copy_mapped_to_plotted(mapped_decimal_storage_type const* value)
{
	switch (_decimal_type)
	{
		case PVCore::UnsignedIntegerType:
		{
			// Direct copy. Vectorized by GCC
			uint32_t const* const vint = &value->storage_as_uint();
			for (size_t i = 0; i < _dest_size; i++) {
				_dest[i] = vint[i];
			}
			break;
		}
		case PVCore::IntegerType:
		{
			// Change signed integer so that -2**31 is zero.
			// TODO: check that GCC vectorize this!
			uint32_t const* const vint = &value->storage_as_uint();
			for (size_t i = 0; i < _dest_size; i++) {
				const uint32_t v = vint[i];
				_dest[i] = ((~v) & 0x80000000) | (v & 0x7FFFFFFF);
			}
			break;
		}
		case PVCore::FloatType:
		{
			// Pretty basic for now, and not really interesting..
			float const* const vfloat = &value->storage_as_float();
			// That should also be vectorized!
			for (size_t i = 0; i < _dest_size; i++) {
				_dest[i] = (uint32_t) vfloat[i];
			}
			break;
		}
		default:
			assert(false);
			break;
	}
}
