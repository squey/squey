//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <inendi/PVAxis.h>          // for PVAxis
#include <inendi/PVMappingFilter.h> // for PVMappingFilter, etc
#include <inendi/PVMappingProperties.h>

#include <pvkernel/rush/PVFormat.h> // for PVFormat

#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS, etc
#include <pvkernel/core/PVCompList.h>     // for comp_hash
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject

#include <QList>   // for QList
#include <QString> // for QString

#include <memory> // for __shared_ptr

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVFormat const& format, PVCol idx)
    : PVMappingProperties(format.get_axes().at(idx))
{
}

Inendi::PVMappingProperties::PVMappingProperties(PVRush::PVAxisFormat const& axis_format)
    : PVMappingProperties(Inendi::PVAxis(axis_format).get_mapping().toStdString(),
                          Inendi::PVAxis(axis_format).get_args_mapping())
{
}

Inendi::PVMappingProperties::PVMappingProperties(std::string mode, PVCore::PVArgumentList args)
    : _mode(std::move(mode))
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
	auto mode = so.attribute_read<QString>("mode");

	PVCore::PVArgumentList args;
	so.arguments_read("properties", args, args);
	return {mode.toStdString(), args};
}

void Inendi::PVMappingProperties::serialize_write(PVCore::PVSerializeObject& so) const
{
	QString mode = QString::fromStdString(_mode);
	so.attribute_write("mode", mode);
	so.arguments_write("properties", _args);
}
