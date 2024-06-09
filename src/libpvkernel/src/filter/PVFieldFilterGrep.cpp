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

#include <pvkernel/filter/PVFieldFilterGrep.h> // for PVFieldFilterGrep
#include <pvkernel/filter/PVFieldsFilter.h>    // for PVFieldsFilter, etc
#include <pvkernel/filter/PVFilterFunction.h>  // for DEFAULT_ARGS_FILTER, etc
#include <pvkernel/core/PVArgument.h> // for PVArgumentList
#include <pvkernel/core/PVElement.h>  // for list_fields, PVElement
#include <pvkernel/core/PVField.h>    // for PVField
#include <pvkernel/core/PVOrderedMap.h> // for PVOrderedMap
#include <QString>  // for QString
#include <QVariant> // for QVariant

/******************************************************************************
 *
 * PVFilter::PVFieldFilterGrep::PVFieldFilterGrep
 *
 *****************************************************************************/
PVFilter::PVFieldFilterGrep::PVFieldFilterGrep(PVCore::PVArgumentList const& args)
    : PVFilter::PVFieldsFilter<PVFilter::one_to_one>()
{
	INIT_FILTER(PVFilter::PVFieldFilterGrep, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterGrep)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterGrep)
{
	PVCore::PVArgumentList args;
	args["str"] = QString();
	args["reverse"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterGrep::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldFilterGrep::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_str = _args.at("str").toString();
	_inverse = args.at("reverse").toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterGrep::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldFilterGrep::one_to_one(PVCore::PVField& obj)
{
	QString str = QString::fromLatin1(obj.begin(), obj.size());
	bool found = str.contains(_str);
	if (!(found ^ _inverse)) {
		obj.set_filtered();
		obj.elt_parent()->set_filtered();
	}
	return obj;
}
