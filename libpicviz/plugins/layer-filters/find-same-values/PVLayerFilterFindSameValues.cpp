//! \file PVLayerFilterFindSameValues.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QSpinBox>

#include "PVLayerFilterFindSameValues.h"

#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVSpinBoxType.h>
#include <pvkernel/rush/PVUtils.h>
#include <picviz/PVView.h>

#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis"

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindSameValues::PVLayerFilterFindSameValues
 *
 *****************************************************************************/
Picviz::PVLayerFilterFindSameValues::PVLayerFilterFindSameValues(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterFindSameValues, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindSameValues)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindSameValues)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("axes", "Axes")].setValue(PVCore::PVAxesIndexType());

	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindSameValues::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterFindSameValues::get_default_args_for_view(PVView const& view)
{
	// Retrieve the key axes of the PVFormat of that PVView
	PVCore::PVArgumentList args = get_args();
	args["axes"].setValue(PVCore::PVAxesIndexType(view.get_original_axes_index_with_tag(get_tag("key"))));
	return args;
}


/******************************************************************************
 *
 * Picviz::PVLayerFilterFindSameValues::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterFindSameValues::operator()(PVLayer& in, PVLayer &out)
{	
	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	PVRow nb_lines = nraw.get_nrows();
	PVCore::PVAxesIndexType axes = _args["axes"].value<PVCore::PVAxesIndexType>();
	if (axes.size() == 0) {
		_args = get_default_args_for_view(*_view);
		axes = _args["axes"].value<PVCore::PVAxesIndexType>();
		if (axes.size() == 0) {
			PVLOG_ERROR("(PVLayerFilterFindSameValues) no key axes defined in the format and no axes selected !\n");
			if (&in != &out) {
				out = in;
			}
			return;
		}
	}

	out.get_selection().select_none();

	QHash<QString,int> lines_hash;
	QString key;
	PVRow counter;

	/* 1st round: we create our hash from the values */
	for (counter = 0; counter < nb_lines; counter++) {
		PVRush::PVNraw::const_nraw_table_line nrawvalues = nraw.get_row(counter);
		if (in.get_selection().get_line(counter)) {
			key = PVRush::PVUtils::generate_key_from_axes_values(axes, nrawvalues);
			lines_hash.insert(key, 1);
		}
	}


	// /* 2nd round: we get the color from the ratio compared with the key and the frequency */
	for (counter = 0; counter < nb_lines; counter++) {
		PVRush::PVNraw::const_nraw_table_line nrawvalues = nraw.get_row(counter);

		key = PVRush::PVUtils::generate_key_from_axes_values(axes, nrawvalues);

		int value_from_key = lines_hash[key];
		if (value_from_key > 0) { 
			out.get_selection().set_line(counter, true);
		}
	}

}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterFindSameValues::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_args_for_preset().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

IMPL_FILTER(Picviz::PVLayerFilterFindSameValues)
