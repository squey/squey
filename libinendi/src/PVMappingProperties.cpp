/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVSource.h>
#include <inendi/PVMappingProperties.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVFormatVersion.h>

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVFormat const& format, PVCol idx)
    : PVMappingProperties(format.get_axes().at(idx), idx)
{
}

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVAxisFormat const& axis_format, PVCol idx)
    : PVMappingProperties(Inendi::PVAxis(axis_format).get_mapping().toStdString(),
                          Inendi::PVAxis(axis_format).get_args_mapping(),
                          idx)
{
}

Inendi::PVMappingProperties::PVMappingProperties(std::string const& mode,
                                                 PVCore::PVArgumentList args,
                                                 PVCol idx)
    : _index(idx)
    , _mapping_filter(LIB_CLASS(Inendi::PVMappingFilter)::get()
                          .get_class_by_name(QString::fromStdString(_mode))
                          ->clone<PVMappingFilter>())
    , _mode(mode)
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

bool Inendi::PVMappingProperties::operator==(const PVMappingProperties& org)
{
	// These properties are equal if and only if the same filter is used on the
	// same index
	return (_mapping_filter == org._mapping_filter) && (_index == org._index);
}

Inendi::PVMappingProperties
Inendi::PVMappingProperties::serialize_read(PVCore::PVSerializeObject& so,
                                            Inendi::PVMapping const& parent)
{
	PVCol idx;
	so.attribute("index", idx);

	QString mode;
	so.attribute("mode", mode);

	if (so.get_version() <= 2) {
		QString type =
		    parent.get_mapped()->get_parent().get_rushnraw().collection().formatter(idx)->name();
		mode = PVRush::PVFormatVersion::get_mapped_from_format(type, mode);
	}

	PVCore::PVArgumentList args;
	so.arguments("properties", args, args);
	return {mode.toStdString(), args, idx};
}

void Inendi::PVMappingProperties::serialize_write(PVCore::PVSerializeObject& so)
{
	so.attribute("index", _index);
	QString mode = QString::fromStdString(_mode);
	so.attribute("mode", mode);
	so.arguments("properties", _args, _mapping_filter->get_default_args());
}

void Inendi::PVMappingProperties::set_default_args(PVRush::PVAxisFormat const& axis)
{
	if (_args.size() == 0) {
		set_args(Inendi::PVAxis(axis).get_args_mapping());
	}
}
