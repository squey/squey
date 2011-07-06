//! \file PVLayerFilterDiff.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QSpinBox>

#include "PVLayerFilterDiff.h"
#include <picviz/PVColor.h>
#include <picviz/PVView.h>
#include <pvcore/PVAxesIndexType.h>
#include <pvcore/PVSpinBoxType.h>

static QString generate_row_key_from_values(PVCore::PVAxesIndexType const& axes, PVRush::PVNraw::nraw_table_line const& values)
{
	QString ret;
	PVCore::PVAxesIndexType::const_iterator it;
	for (it = axes.begin(); it != axes.end(); it++) {
		ret.append(values[*it]);
	}
	return ret;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterDiff::PVLayerFilterDiff
 *
 *****************************************************************************/
Picviz::PVLayerFilterDiff::PVLayerFilterDiff(PVFilter::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterDiff, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterDiff)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterDiff)
{
	PVFilter::PVArgumentList args;
	args["Axes"].setValue(PVCore::PVAxesIndexType());
	args["From line"].setValue(PVCore::PVSpinBoxType());
	args["To line"].setValue(PVCore::PVSpinBoxType());

	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterDiff::get_default_args_for_view
 *
 *****************************************************************************/
PVFilter::PVArgumentList Picviz::PVLayerFilterDiff::get_default_args_for_view(PVView const& view)
{
	// Retrieve the key axes of the PVFormat of that PVView
	PVFilter::PVArgumentList args = get_args();
	PVCore::PVAxesIndexType key_axes;
	PVRush::PVFormat::list_axes const& axes = view.get_source_parent()->nraw->format->axes;
	PVRush::PVFormat::list_axes::const_iterator it;
	int axis_id = 0;
	// FIXME:
	// In PVAxesCombination::set_from_format, the id are computed like that
	// It might be safer to use PVAxesCombination to get this information
	for (it = axes.begin(); it != axes.end(); it++) {
		QHash<QString,QString> const& params = *it;
		if (params["key"].compare("true", Qt::CaseInsensitive) == 0) {
			key_axes.push_back(axis_id);
		}
		axis_id++;
	}
	args["Axes"].setValue(key_axes);
	return args;
}


/******************************************************************************
 *
 * Picviz::PVLayerFilterDiff::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterDiff::operator()(PVLayer& in, PVLayer &out)
{	
	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	PVRow nb_lines = nraw.size();
	PVCore::PVAxesIndexType axes = _args["Axes"].value<PVCore::PVAxesIndexType>();
	if (axes.size() == 0) {
		_args = get_default_args_for_view(*_view);
		axes = _args["Axes"].value<PVCore::PVAxesIndexType>();
		if (axes.size() == 0) {
			PVLOG_ERROR("(PVLayerFilterHeatlineBase) no key axes defined in the format and no axes selected !\n");
			if (&in != &out) {
				out = in;
			}
			return;
		}
	}

	PVCore::PVSpinBoxType fromline_spinbox = _args["From line"].value<PVCore::PVSpinBoxType>();
	PVCore::PVSpinBoxType toline_spinbox = _args["To line"].value<PVCore::PVSpinBoxType>();

	// PVLOG_INFO("************************************* From line:%d\n", fromline_spinbox.get_value());
	// PVLOG_INFO("************************************* To line:%d\n", toline_spinbox.get_value());
	// qDebug << "from line= " << fromline_spinbox.value();

	out.get_selection().select_all();

	QHash<QString,int> lines_hash;
	QString key;
	PVRow counter;

	/* 1st round: we create our hash from the values */
	for (counter = 0; counter < toline_spinbox.get_value(); counter++) {
		PVRush::PVNraw::nraw_table_line const& nrawvalues = nraw.at(counter);
		key = generate_row_key_from_values(axes, nrawvalues);
		// qDebug("KEY VALUE=%s\n", qPrintable(key));

		lines_hash.insert(key, 1);
	}


	// /* 2nd round: we get the color from the ratio compared with the key and the frequency */
	for (counter = 0; counter < nb_lines; counter++) {
		PVRush::PVNraw::nraw_table_line const& nrawvalues = nraw.at(counter);
		key = generate_row_key_from_values(axes, nrawvalues);
		int value_from_key = lines_hash[key];
		if (value_from_key == 1) { 
			out.get_selection().set_line(counter, 0);
		}
	}

}

IMPL_FILTER(Picviz::PVLayerFilterDiff)
