/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVSelectionFilter.h>
#include <assert.h>



/******************************************************************************
 *
 * Inendi::PVSelectionFilter::PVSelectionFilter
 *
 *****************************************************************************/
Inendi::PVSelectionFilter::PVSelectionFilter(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(Inendi::PVSelectionFilter, args);
	set_output(NULL);
	_view = NULL;
}

/******************************************************************************
 *
 * Inendi::PVSelectionFilter::
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Inendi::PVSelectionFilter)
{
	return PVCore::PVArgumentList();
}

/******************************************************************************
 *
 * Inendi::PVSelectionFilter::operator_differentout
 *
 *****************************************************************************/
Inendi::PVSelection& Inendi::PVSelectionFilter::operator_differentout(PVSelection &selection)
{
	operator()(selection, *_out_p);
	return *_out_p;
}

/******************************************************************************
 *
 * Inendi::PVSelectionFilter::operator()
 *
 *****************************************************************************/
Inendi::PVSelection& Inendi::PVSelectionFilter::operator()(PVSelection& selection)
{
	assert(_view);
	if (_out_p)
		return operator_differentout(selection);
	return operator_sameout(selection);
}

/******************************************************************************
 *
 * Inendi::PVSelectionFilter::operator()
 *
 *****************************************************************************/
void Inendi::PVSelectionFilter::operator()(PVSelection &in, PVSelection &out)
{
	// By default, if out != in, copy it
	if (&out != &in)
		out = in;
}

/******************************************************************************
 *
 * Inendi::PVSelectionFilter::operator_sameout
 *
 *****************************************************************************/
Inendi::PVSelection& Inendi::PVSelectionFilter::operator_sameout(PVSelection &selection)
{
	operator()(selection, selection);
	return selection;
}

/******************************************************************************
 *
 * Inendi::PVSelectionFilter::set_output
 *
 *****************************************************************************/
void Inendi::PVSelectionFilter::set_output(PVSelection* out)
{
	_out_p = out;

/*	AG: do not remember the clean way to do this... later
 	if (out == NULL)
		this->operator()(PVSelection&) = &(this->operator_sameout);
	else
		this->operator()(PVSelection&) = &(this->operator_differentout);*/
}

/******************************************************************************
 *
 * Inendi::PVSelectionFilter::set_view
 *
 *****************************************************************************/
void Inendi::PVSelectionFilter::set_view(PVView const& view)
{
	_view = &view;
}

IMPL_FILTER(Inendi::PVSelectionFilter)
