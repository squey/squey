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

#include <inendi/PVAxis.h>               // for PVAxis
#include <inendi/PVPlottingFilter.h>     // for PVPlottingFilter, etc
#include <inendi/PVPlottingProperties.h> // for PVPlottingProperties

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat
#include <pvkernel/rush/PVFormat.h>     // for PVFormat

#include <pvkernel/core/PVArgument.h>     // for PVArgumentList, etc
#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS, etc
#include <pvkernel/core/PVCompList.h>     // for comp_hash
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVSerializeObject.h> // for PVSerializeObject

#include "pvbase/types.h" // for PVCol

#include <QList>   // for QList
#include <QString> // for QString

#include <memory> // for __shared_ptr
#include <string> // for string, operator==, etc

/******************************************************************************
 *
 * Inendi::PVPlottingProperties::PVPlottingProperties
 *
 *****************************************************************************/
Inendi::PVPlottingProperties::PVPlottingProperties(PVRush::PVFormat const& format, PVCol idx)
    : PVPlottingProperties(format.get_axes().at(idx))
{
}

Inendi::PVPlottingProperties::PVPlottingProperties(PVRush::PVAxisFormat const& axis_format)
    : PVPlottingProperties(Inendi::PVAxis(axis_format).get_plotting().toStdString(),
                           Inendi::PVAxis(axis_format).get_args_plotting())
{
}

Inendi::PVPlottingProperties::PVPlottingProperties(std::string mode, PVCore::PVArgumentList args)
    : _mode(std::move(mode))
    , _plotting_filter(LIB_CLASS(Inendi::PVPlottingFilter)::get()
                           .get_class_by_name(QString::fromStdString(_mode))
                           ->clone<PVPlottingFilter>())
    , _is_uptodate(false)
{
	set_args(args);
}

void Inendi::PVPlottingProperties::set_args(PVCore::PVArgumentList const& args)
{
	if (!PVCore::comp_hash(_args, args)) {
		_is_uptodate = false;
	}
	if (_plotting_filter) {
		PVCore::PVArgumentList new_args = _plotting_filter->get_default_args();
		PVArgumentList_set_common_args_from(new_args, args);
		_plotting_filter->set_args(new_args);
		_args = new_args;
	} else {
		_args = args;
	}
}

void Inendi::PVPlottingProperties::set_mode(std::string const& mode)
{
	if (_is_uptodate) {
		_is_uptodate = (_mode == mode);
	}

	_mode = mode;
	PVPlottingFilter::p_type lib_filter =
	    LIB_CLASS(PVPlottingFilter)::get().get_class_by_name(QString::fromStdString(mode));

	_plotting_filter = lib_filter->clone<PVPlottingFilter>();
	set_args(_args);
}

Inendi::PVPlottingFilter::p_type Inendi::PVPlottingProperties::get_plotting_filter()
{
	return _plotting_filter;
}

Inendi::PVPlottingProperties
Inendi::PVPlottingProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	auto mode = so.attribute_read<QString>("mode");

	PVCore::PVArgumentList args;
	so.arguments_read("properties", args, args);
	return {mode.toStdString(), args};
}

void Inendi::PVPlottingProperties::serialize_write(PVCore::PVSerializeObject& so) const
{
	QString mode = QString::fromStdString(_mode);
	so.attribute_write("mode", mode);
	so.arguments_write("properties", _args);
}
