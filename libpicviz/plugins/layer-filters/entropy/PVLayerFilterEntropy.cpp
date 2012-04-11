//! \file PVLayerFilterEntropy.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2012
//! Copyright (C) Philippe Saadé 2009-2012
//! Copyright (C) Picviz Labs 2011-2012

#include "PVLayerFilterEntropy.h"

#include <pvkernel/rush/PVAxisTagsDec.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <picviz/PVView.h>

#include <QDir>
#include <QList>
#include <QString>

#include <math.h>

#define ARG_NAME_AXIS "axis"
#define ARG_DESC_AXIS "Axis From"
#define ARG_NAME_NUMBER "number"
#define ARG_DESC_NUMBER "Number"

/******************************************************************************
 *
 * Picviz::PVLayerFilterEntropy::PVLayerFilterEntropy
 *
 *****************************************************************************/
Picviz::PVLayerFilterEntropy::PVLayerFilterEntropy(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l), rules_number(0)
{
	INIT_FILTER(PVLayerFilterEntropy, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterEntropy)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterEntropy)
{
	PVCore::PVArgumentList args;

	args[PVCore::PVArgumentKey(ARG_NAME_AXIS, QObject::tr(ARG_DESC_AXIS))].setValue(PVCore::PVAxisIndexType(0));
	args[PVCore::PVArgumentKey(ARG_NAME_NUMBER, QObject::tr(ARG_DESC_NUMBER))] = QString("4");

	return args;
}

float get_entropy(QString str)
{
	float entropy = 0;

	if (str.isEmpty()) {
		return 0;
	}

	for(int i=0; i < 256; i++) {
		float occurences = (float) ((float)str.count(QChar(i))) / str.length();
		if (occurences > 0) {
			entropy += - occurences*(log2f(occurences));
		}
	}

	return entropy;
// def H(data):
//     if not data:
// 	return 0

//     entropy = 0
//     for x in range(256):
// 	p_x = float(data.count(chr(x)))/len(data)
//         if p_x > 0:
//         	entropy += - p_x*math.log(p_x, 2)

//     return entropy

// ratio = logf(count_frequency) / logf(highest_frequency);

}

/******************************************************************************
 *
 * Picviz::PVLayerFilterEntropy::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterEntropy::operator()(PVLayer& in, PVLayer &out)
{
	int axis_id = _args[ARG_NAME_AXIS].value<PVCore::PVAxisIndexType>().get_original_index();
	QString number = _args[ARG_NAME_NUMBER].toString();
	float greater_counter = number.toFloat();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	PVRow nb_lines = _view->get_qtnraw_parent().get_nrows();

	out.get_selection().select_none();

	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		if (_view->get_line_state_in_pre_filter_layer(r)) {
			PVRush::PVNraw::const_nraw_table_line nraw_r = nraw.get_row(r);
			float entr = get_entropy(nraw_r[axis_id].get_qstr());
			if (entr > greater_counter) {
				out.get_selection().set_line(r, true);
			}
			// PVLOG_INFO("Entropy:%f\n", entr);
			// bool sel = !((re.indexIn(nraw_r[axis_id].get_qstr()) != -1) ^ include);
			// out.get_selection().set_line(r, sel);
		}
	}

}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterEntropy::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_default_args().keys();
	keys.removeAll(ARG_NAME_AXIS);
	return keys;
}

QString Picviz::PVLayerFilterEntropy::status_bar_description()
{
	return QString("Search strings matching for Shannon Entropy.");
}

QString Picviz::PVLayerFilterEntropy::detailed_description()
{
	return QString("<b>Purpose</b><br/>This filter select strings using Shannon entropy<hr><b>Behavior</b><br/>It will select any string that are greater than the wanted entropy. To find obcure strings, it is advised to search for something greater than 4.");
}

IMPL_FILTER(Picviz::PVLayerFilterEntropy)
