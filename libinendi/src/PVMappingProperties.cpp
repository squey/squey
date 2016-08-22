/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMappingProperties.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVFormatVersion.h>

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVFormat const& format, PVCol idx)
    : PVMappingProperties(format.get_axes().at(idx))
{
}

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVAxisFormat const& axis_format)
    : PVMappingProperties(Inendi::PVAxis(axis_format).get_mapping().toStdString(),
                          Inendi::PVAxis(axis_format).get_args_mapping())
{
}

Inendi::PVMappingProperties::PVMappingProperties(std::string const& mode,
                                                 PVCore::PVArgumentList args)
    : _mode(mode)
    , _mapping_filter(LIB_CLASS(Inendi::PVMappingFilter)::get()
                          .get_class_by_name(QString::fromStdString(_mode))
                          ->clone<PVMappingFilter>())
    , _is_uptodate(false)
{
	set_args(args);
}

void Inendi::PVMappingProperties::set_args(PVCore::PVArgumentList const& args)
{
	if (!PVCore::comp_hash(args, _args)) {
		_is_uptodate = false;
	}
	if (_mapping_filter) {
		PVCore::PVArgumentList new_args = _mapping_filter->get_default_args();
		PVArgumentList_set_common_args_from(new_args, args);
		_mapping_filter->set_args(new_args);
		_args = new_args;
	} else {
		_args = args;
	}
}

void Inendi::PVMappingProperties::set_mode(std::string const& mode)
{
	if (_is_uptodate && _mode == mode) {
		return;
	}
	_is_uptodate = false;
	_mode = mode;
	PVMappingFilter::p_type lib_filter =
	    LIB_CLASS(Inendi::PVMappingFilter)::get().get_class_by_name(QString::fromStdString(mode));

	_mapping_filter = lib_filter->clone<PVMappingFilter>();
	set_args(_args);
}

Inendi::PVMappingProperties
Inendi::PVMappingProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	QString mode;
	so.attribute("mode", mode);

	PVCore::PVArgumentList args;
	so.arguments("properties", args, args);
	return {mode.toStdString(), args};
}

void Inendi::PVMappingProperties::serialize_write(PVCore::PVSerializeObject& so)
{
	QString mode = QString::fromStdString(_mode);
	so.attribute("mode", mode);
	so.arguments("properties", _args, _mapping_filter->get_default_args());
}
