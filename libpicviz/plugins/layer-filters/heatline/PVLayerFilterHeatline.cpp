//! \file PVLayerFilterHeatlineBase.cpp
//! $Id: PVLayerFilterHeatlineBase.cpp 2526 2011-05-02 12:21:26Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterHeatline.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVUtils.h>
#include <picviz/PVView.h>

#include <math.h>

/******************************************************************************
 *
 * Picviz::PVLayerFilterHeatlineBase::PVLayerFilterHeatlineBase
 *
 *****************************************************************************/
Picviz::PVLayerFilterHeatlineBase::PVLayerFilterHeatlineBase(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterHeatlineBase, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineBase)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineBase)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("axes", "Axes")].setValue(PVCore::PVAxesIndexType());

	PVCore::PVEnumType scale(QStringList() << "Linear" << "Log", 0);
	args[PVCore::PVArgumentKey("scale", "Scale")].setValue(scale);

	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterHeatlineBase::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterHeatlineBase::get_default_args_for_view(PVView const& view)
{
	PVCore::PVArgumentList args = get_default_args();
	// Default args with the "key" tag
	args["axes"].setValue(PVCore::PVAxesIndexType(view.get_original_axes_index_with_tag(get_tag("key"))));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterHeatlineBase::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterHeatlineBase::operator()(PVLayer& in, PVLayer &out)
{	
	PVRow nb_lines;
	PVRow counter;
	PVRow highest_frequency;
	QString key;
	float ratio;
	
	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	nb_lines = nraw.get_nrows();
	
	PVCore::PVAxesIndexType axes = _args.value("axes").value<PVCore::PVAxesIndexType>();
	if (axes.size() == 0) {
		_args = get_default_args_for_view(*_view);
		axes = _args.value("axes").value<PVCore::PVAxesIndexType>();
		if (axes.size() == 0) {
			PVLOG_ERROR("(PVLayerFilterHeatlineBase) no key axes defined in the format and no axes selected !\n");
			if (&in != &out) {
				out = in;
			}
			return;
		}
	}

	bool bLog = _args.value("scale").value<PVCore::PVEnumType>().get_sel().compare("Log") == 0;

	highest_frequency = 1;

	QHash<QString,PVRow> lines_hash;

	out.get_selection() = in.get_selection();

	/* 1st round: we calculate all the frequencies */
	for (counter = 0; counter < nb_lines; counter++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}
		if (!_view->get_line_state_in_pre_filter_layer(counter)) {
			continue;
		}
		PVRush::PVNraw::const_nraw_table_line nrawvalues = nraw.get_row(counter);
		key = PVRush::PVUtils::generate_key_from_axes_values(axes, nrawvalues);

		PVRow count_frequency = lines_hash[key]+1;
		lines_hash.insert(key, count_frequency);
		if (count_frequency > highest_frequency) {
			highest_frequency = count_frequency;
		}
	}

	/* 2nd round: we get the color from the ratio compared with the key and the frequency */
	for (counter = 0; counter < nb_lines; counter++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}
		if (_view->get_line_state_in_pre_filter_layer(counter)) {
			PVRush::PVNraw::const_nraw_table_line nrawvalues = nraw.get_row(counter);
			key = PVRush::PVUtils::generate_key_from_axes_values(axes, nrawvalues);

			PVRow count_frequency = lines_hash[key];

			if (bLog) {
				ratio = logf(count_frequency) / logf(highest_frequency);
			}
			else {
				ratio = (float)count_frequency / (float)highest_frequency;
			}

			this->post(in, out, ratio, counter);
		}
	}
}

void Picviz::PVLayerFilterHeatlineBase::post(PVLayer& /*in*/, PVLayer& /*out*/, float /*ratio*/, PVRow /*line_id*/)
{
	// The base filter does nothing
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineBase)


// "Colorize mode" filter

Picviz::PVLayerFilterHeatlineColor::PVLayerFilterHeatlineColor(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineColor, l);
}

DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineColor)
{
	return Picviz::PVLayerFilterHeatlineBase::default_args();
}

void Picviz::PVLayerFilterHeatlineColor::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	PVCore::PVColor color;
	QColor qcolor;

	qcolor.setHsvF((1.0 - ratio)/3.0, 1.0, 1.0);
	color.fromQColor(qcolor);

	out.get_lines_properties().line_set_rgb_from_color(line_id, color);
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineColor)


// "Selection mode" filter

Picviz::PVLayerFilterHeatlineSel::PVLayerFilterHeatlineSel(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineSel, l);
}

DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineSel)
{
	PVCore::PVArgumentList args = Picviz::PVLayerFilterHeatlineBase::default_args();
	args[PVCore::PVArgumentKey("colors", "Colors")].setValue(PVCore::PVColorGradientDualSliderType());
	return args;
}

void Picviz::PVLayerFilterHeatlineSel::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	PVCore::PVColorGradientDualSliderType ratios = _args["colors"].value<PVCore::PVColorGradientDualSliderType>();

	const float *v = ratios.get_positions();

	float fmin = v[0];
	float fmax = v[1];

	if ((ratio > fmax) || (ratio < fmin)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineSel)

// "Select and colorize" mode
Picviz::PVLayerFilterHeatlineSelAndCol::PVLayerFilterHeatlineSelAndCol(PVCore::PVArgumentList const& l)
	: PVLayerFilterHeatlineBase(l)
{
	INIT_FILTER(PVLayerFilterHeatlineSelAndCol, l);
}

DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
{
	PVCore::PVArgumentList args = Picviz::PVLayerFilterHeatlineBase::default_args();
	args[PVCore::PVArgumentKey("colors", "Colors")].setValue(PVCore::PVColorGradientDualSliderType());
	return args;
}

void Picviz::PVLayerFilterHeatlineSelAndCol::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	// Colorize
	PVCore::PVColor color;
	QColor qcolor;

	qcolor.setHsvF((1.0 - ratio)/3.0, 1.0, 1.0);
	color.fromQColor(qcolor);

	out.get_lines_properties().line_set_rgb_from_color(line_id, color);

	// Select
	PVCore::PVColorGradientDualSliderType ratios = _args.value("colors").value<PVCore::PVColorGradientDualSliderType>();

	const float *v = ratios.get_positions();

	float fmin = v[0];
	float fmax = v[1];

	if ((ratio > fmax) || (ratio < fmin)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
