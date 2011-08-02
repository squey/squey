//! \file PVElementFilterGrep.cpp
//! $Id: PVElementFilterGrep.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/filter/PVElementFilterGrep.h>
#include <assert.h>

/******************************************************************************
 *
 * PVFilter::PVElementFilterGrep::PVElementFilterGrep
 *
 *****************************************************************************/
PVFilter::PVElementFilterGrep::PVElementFilterGrep(PVCore::PVArgumentList const& args) :
	PVElementFilter()
{
	INIT_FILTER(PVFilter::PVElementFilterGrep, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVElementFilterGrep)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVElementFilterGrep)
{
	PVCore::PVArgumentList args;
	args["str"] = QString();
	args["reverse"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVElementFilterGrep::set_args
 *
 *****************************************************************************/
void PVFilter::PVElementFilterGrep::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_str = _args["str"].toString();
	_inverse = args["reverse"].toBool();
}

/******************************************************************************
 *
 * PVFilter::PVElementFilterGrep::operator
 *
 *****************************************************************************/
PVCore::PVElement& PVFilter::PVElementFilterGrep::operator()(PVCore::PVElement& obj)
{
	QString str = QString::fromAscii(obj.begin(), obj.size());
	bool found = str.contains(_str);
	if (!(found ^ _inverse))
		obj.set_invalid();
	return obj;
}

IMPL_FILTER(PVFilter::PVElementFilterGrep)
