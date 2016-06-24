/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>

#include <inendi/PVPlottingFilter.h>

Inendi::PVPlottingFilter::PVPlottingFilter()
    : PVFilter::PVFilterFunctionBase<uint32_t*, pvcop::db::array const&>()
    , PVCore::PVRegistrableClass<Inendi::PVPlottingFilter>()
{
	_dest = NULL;
	_dest_size = 0;
}

void Inendi::PVPlottingFilter::set_mapping_mode(QString const& mode)
{
	_mapping_mode = mode;
}

void Inendi::PVPlottingFilter::set_dest_array(PVRow size, uint32_t* array)
{
	assert(array);
	_dest_size = size;
	_dest = array;
}

QStringList Inendi::PVPlottingFilter::list_modes(QString const& type, bool only_expandable)
{
	LIB_CLASS(PVPlottingFilter)
	::list_classes const& pl_filters = LIB_CLASS(PVPlottingFilter)::get().get_list();
	LIB_CLASS(PVPlottingFilter)::list_classes::const_iterator it;
	QStringList ret;
	for (it = pl_filters.begin(); it != pl_filters.end(); it++) {
		if (only_expandable && !it->value()->can_expand()) {
			continue;
		}
		QString const& name = it->key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << params[1];
		}
	}
	return ret;
}

QList<Inendi::PVPlottingFilter::p_type>
Inendi::PVPlottingFilter::list_modes_lib(QString const& type, bool only_expandable)
{
	LIB_CLASS(PVPlottingFilter)
	::list_classes const& pl_filters = LIB_CLASS(PVPlottingFilter)::get().get_list();
	LIB_CLASS(PVPlottingFilter)::list_classes::const_iterator it;
	QList<p_type> ret;
	ret.reserve(pl_filters.size());
	for (it = pl_filters.begin(); it != pl_filters.end(); it++) {
		if (only_expandable && !it->value()->can_expand()) {
			continue;
		}
		QString const& name = it->key();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			ret << it->value();
		}
	}
	return ret;
}

QString Inendi::PVPlottingFilter::mode_from_registered_name(QString const& rn)
{
	QStringList params = rn.split('_');
	return params[1];
}

QString Inendi::PVPlottingFilter::get_human_name() const
{
	return mode_from_registered_name(registered_name());
}

void Inendi::PVPlottingFilter::copy_mapped_to_plotted(pvcop::db::array const& mapped)
{
	switch (mapped.type()) {
	case pvcop::db::type_uint32: {
		auto& mapped_array = mapped.to_core_array<uint32_t>();
		// Direct copy. Vectorized by GCC
		for (size_t i = 0; i < _dest_size; i++) {
			_dest[i] = ~mapped_array[i];
		}
		break;
	}
	case pvcop::db::type_int32: {
		auto& mapped_array = mapped.to_core_array<int32_t>();
		// Change signed integer so that -2**31 is zero.
		// TODO: check that GCC vectorize this!
		for (size_t i = 0; i < _dest_size; i++) {
			const uint32_t v = mapped_array[i];
			_dest[i] = ~(((~v) & 0x80000000) | (v & 0x7FFFFFFF));
		}
		break;
	}
	case pvcop::db::type_float: {
		auto& mapped_array = mapped.to_core_array<float>();
		// Pretty basic for now, and not really interesting..
		// That should also be vectorized!
		for (size_t i = 0; i < _dest_size; i++) {
			_dest[i] = ~((uint32_t)mapped_array[i]);
		}
		break;
	}
	default:
		assert(false);
		break;
	}
}
