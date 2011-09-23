//! \file PVLayerFilterWebmailFinder.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterWebmailFinder.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

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
	//int axis_id = _args["Domain Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	QList<PVCol> axes_id = _view->get_original_axes_index_with_tag(get_tag("domain"));

	PVRow nb_lines = _view->get_qtnraw_parent().size();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();

	PVSelection hotmail_sel;
	PVLinesProperties hotmail_lp;
	QString hotmail("mail.live.com");

	QString yahoo("mail.yahoo.com");
	PVSelection yahoo_sel;
	PVLinesProperties yahoo_lp;

	PVLOG_DEBUG("(Picviz::PVLayerFilterWebmailFinder) %d axes with the tag domain.\n", axes_id.size());
	for (int i = 0; i < axes_id.size(); i++) {
		int axis_id = axes_id[i];
		PVLOG_DEBUG("(Picviz::PVLayerFilterWebmailFinder) process with axis '%d'.\n", axis_id);
		
		for (PVRow r = 0; r < nb_lines; r++) {
			if (should_cancel()) {
				if (&in != &out) {
					out = in;
				}
				return;
			}
			if (_view->get_line_state_in_pre_filter_layer(r)) {
				PVRush::PVNraw::nraw_table_line const& nraw_r = nraw.at(r);
				QString const& data = nraw_r.at(axis_id);
				hotmail_sel.set_line(r, data.contains(hotmail, Qt::CaseInsensitive));
				yahoo_sel.set_line(r, data.contains(yahoo, Qt::CaseInsensitive));
			}
		}
	}

	PVLayer yahoo_layer("Yahoo", yahoo_sel, yahoo_lp);
	_view->layer_stack.append_layer(yahoo_layer);

	PVLayer hotmail_layer("Hotmail", hotmail_sel, hotmail_lp);
	_view->layer_stack.append_layer(hotmail_layer);
}

IMPL_FILTER(Picviz::PVLayerFilterWebmailFinder)
