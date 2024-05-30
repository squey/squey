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

#include <squey/PVAxis.h>               // for PVAxis
#include <squey/PVScalingFilter.h>     // for PVScalingFilter, etc
#include <squey/PVScalingProperties.h> // for PVScalingProperties

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
 * Squey::PVScalingProperties::PVScalingProperties
 *
 *****************************************************************************/
Squey::PVScalingProperties::PVScalingProperties(PVRush::PVFormat const& format, PVCol idx)
    : PVScalingProperties(format.get_axes().at(idx))
{
}

Squey::PVScalingProperties::PVScalingProperties(PVRush::PVAxisFormat const& axis_format)
    : PVScalingProperties(Squey::PVAxis(axis_format).get_scaling().toStdString(),
                           Squey::PVAxis(axis_format).get_args_scaling())
{
}

Squey::PVScalingProperties::PVScalingProperties(std::string mode, PVCore::PVArgumentList args)
    : _mode(std::move(mode))
    , _scaling_filter(LIB_CLASS(Squey::PVScalingFilter)::get()
                           .get_class_by_name(QString::fromStdString(_mode))
                           ->clone<PVScalingFilter>())
     
{
	set_args(args);
}

void Squey::PVScalingProperties::set_args(PVCore::PVArgumentList const& args)
{
	if (!PVCore::comp_hash(_args, args)) {
		_is_uptodate = false;
	}
	if (_scaling_filter) {
		PVCore::PVArgumentList new_args = _scaling_filter->get_default_args();
		PVArgumentList_set_common_args_from(new_args, args);
		_scaling_filter->set_args(new_args);
		_args = new_args;
	} else {
		_args = args;
	}
}

void Squey::PVScalingProperties::set_mode(std::string const& mode)
{
	if (_is_uptodate) {
		_is_uptodate = (_mode == mode);
	}

	_mode = mode;
	PVScalingFilter::p_type lib_filter =
	    LIB_CLASS(PVScalingFilter)::get().get_class_by_name(QString::fromStdString(mode));

	_scaling_filter = lib_filter->clone<PVScalingFilter>();
	set_args(_args);
}

Squey::PVScalingFilter::p_type Squey::PVScalingProperties::get_scaling_filter()
{
	return _scaling_filter;
}

Squey::PVScalingProperties
Squey::PVScalingProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	auto mode = so.attribute_read<QString>("mode");

	PVCore::PVArgumentList args;
	so.arguments_read("properties", args, args);
	return {mode.toStdString(), args};
}

void Squey::PVScalingProperties::serialize_write(PVCore::PVSerializeObject& so) const
{
	QString mode = QString::fromStdString(_mode);
	so.attribute_write("mode", mode);
	so.arguments_write("properties", _args);
}
