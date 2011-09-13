//! \file PVLayerFilterWebmailFinder.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterWebmailFinder.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>

/******************************************************************************
 *
 * Picviz::PVLayerFilterWebmailFinder::PVLayerFilterWebmailFinder
 *
 *****************************************************************************/
Picviz::PVLayerFilterWebmailFinder::PVLayerFilterWebmailFinder(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterWebmailFinder, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterWebmailFinder)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterWebmailFinder)
{
	PVCore::PVArgumentList args;
	// args["Regular expression"] = QRegExp("(.*)");
	args["Domain Axis"].setValue(PVCore::PVAxisIndexType(0));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterWebmailFinder::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterWebmailFinder::operator()(PVLayer& in, PVLayer &out)
{	
	int axis_id = _args["Domain Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	// QRegExp re = _args["Regular expression"].toRegExp();
	// PVLOG_INFO("Apply filter search to axis %d with regexp %s.\n", axis_id, qPrintable(re.pattern()));

	PVRow nb_lines = _view->get_qtnraw_parent().size();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();

	// Find for hotmail
	QRegExp hotmail_re(".*mail.live.com.*");
	PVSelection hotmail_sel;
	PVLinesProperties hotmail_lp;
	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}
		if (_view->get_line_state_in_pre_filter_layer(r)) {
			PVRush::PVNraw::nraw_table_line const& nraw_r = nraw.at(r);
			hotmail_sel.set_line(r, hotmail_re.indexIn(nraw_r[axis_id]) != -1);
		}
	}
	PVLayer hotmail_layer("Hotmail", hotmail_sel, hotmail_lp);
	_view->layer_stack.append_layer(hotmail_layer);

	QRegExp yahoo_re(".*mail.yahoo.com.*");
	PVSelection yahoo_sel;
	PVLinesProperties yahoo_lp;
	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}
		if (_view->get_line_state_in_pre_filter_layer(r)) {
			PVRush::PVNraw::nraw_table_line const& nraw_r = nraw.at(r);
			yahoo_sel.set_line(r, yahoo_re.indexIn(nraw_r[axis_id]) != -1);
		}
	}
	PVLayer yahoo_layer("Yahoo", yahoo_sel, yahoo_lp);
	_view->layer_stack.append_layer(yahoo_layer);

}

IMPL_FILTER(Picviz::PVLayerFilterWebmailFinder)
