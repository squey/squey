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

#include <inendi/PVAxis.h>
#include <inendi/PVMappingFilter.h>  // for PVMappingFilter, etc
#include <inendi/PVPlottingFilter.h> // for PVPlottingFilter, etc

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat::node_args_t, etc

#include <pvkernel/core/PVArgument.h>     // for PVArgumentList, etc
#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS, etc
#include <pvkernel/core/PVOrderedMap.h>   // for PVOrderedMap
#include <pvkernel/core/PVRegistrableClass.h>

#include <QHash>    // for QHash<>::const_iterator
#include <QString>  // for QString
#include <QVariant> // for QVariant

#include <memory>        // for allocator, __shared_ptr
#include <set>           // for set, etc
#include <string>        // for basic_string, operator+, etc
#include <unordered_set> // for operator==, unordered_set, etc
#include <utility>       // for make_pair, move

/******************************************************************************
 *
 * Inendi::PVAxis::PVAxis
 *
 *****************************************************************************/
Inendi::PVAxis::PVAxis(PVRush::PVAxisFormat axis_format)
    : PVRush::PVAxisFormat(std::move(axis_format))
{
	// Create mapping arguments

	// Get the mapping filter from the library
	{
		Inendi::PVMappingFilter::p_type lib_filter =
		    LIB_CLASS(Inendi::PVMappingFilter)::get().get_class_by_name(get_mapping());
		auto usable_type = lib_filter->list_usable_type();
		if (usable_type.find(get_type().toStdString()) == usable_type.end()) {
			throw Inendi::InvalidPlottingMapping("You can't use mapping :" +
			                                     get_mapping().toStdString() + " with type :" +
			                                     get_type().toStdString());
		}
		PVCore::PVArgumentList def_args = lib_filter->get_default_args();
		_args_mapping = args_from_node(get_args_mapping_string(), def_args);
	}

	// Same for the plotting filter
	{
		Inendi::PVPlottingFilter::p_type lib_filter =
		    LIB_CLASS(Inendi::PVPlottingFilter)::get().get_class_by_name(get_plotting());
		auto usable_type = lib_filter->list_usable_type();
		if (not usable_type.empty() and
		    usable_type.find(std::make_pair(get_type().toStdString(),
		                                    get_mapping().toStdString())) == usable_type.end()) {
			throw Inendi::InvalidPlottingMapping(
			    "You can't use plotting :" + get_plotting().toStdString() + " with mapping :" +
			    get_mapping().toStdString() + " and type :" + get_type().toStdString());
		}
		PVCore::PVArgumentList def_args = lib_filter->get_default_args();
		_args_plotting = args_from_node(get_args_plotting_string(), def_args);
	}
}

PVCore::PVArgumentList Inendi::PVAxis::args_from_node(node_args_t const& args_str,
                                                      PVCore::PVArgumentList const& def_args)
{
	PVCore::PVArgumentList ret;
	node_args_t::const_iterator it;
	for (it = args_str.begin(); it != args_str.end(); ++it) {
		QString const& key(it.key());
		if (def_args.contains(key)) {
			ret[it.key()] = PVCore::QString_to_PVArgument(it.value(), def_args.at(key));
		}
	}
	return ret;
}
