/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldFilterGrep.h>

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
