/**
 * \file PVLayerFilterHeatline.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include "PVLayerFilterHeatline.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/rush/PVUtils.h>
#include <picviz/PVView.h>

#include <pvkernel/core/picviz_bench.h>

#include <math.h>

#include <omp.h>

#define ARG_NAME_AXES "axes"
#define ARG_DESC_AXES "Axes"
#define ARG_NAME_SCALE "scale"
#define ARG_DESC_SCALE "Scale"
#define ARG_NAME_COLORS "colors"
#define ARG_DESC_COLORS "Colors"

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
	args[PVCore::PVArgumentKey(ARG_NAME_AXES, ARG_DESC_AXES)].setValue(PVCore::PVAxesIndexType());

	PVCore::PVEnumType scale(QStringList() << "Linear" << "Log", 0);
	args[PVCore::PVArgumentKey(ARG_NAME_SCALE, ARG_DESC_SCALE)].setValue(scale);

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
	args[ARG_NAME_AXES].setValue(PVCore::PVAxesIndexType(view.get_original_axes_index_with_tag(get_tag("key"))));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterHeatlineBase::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterHeatlineBase::operator()(PVLayer& in, PVLayer &out)
{	
	BENCH_START(heatline);

	PVRow nb_lines;
	PVRow counter;
	PVRow highest_frequency = 1;
	float ratio;
	
	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();
	nb_lines = nraw.get_nrows();
	
	PVCore::PVAxesIndexType axes = _args.value(ARG_NAME_AXES).value<PVCore::PVAxesIndexType>();
	if (axes.size() == 0) {
		_args = get_default_args_for_view(*_view);
		axes = _args.value(ARG_NAME_AXES).value<PVCore::PVAxesIndexType>();
		if (axes.size() == 0) {
			PVLOG_ERROR("(PVLayerFilterHeatlineBase) no key axes defined in the format and no axes selected !\n");
			if (&in != &out) {
				out = in;
			}
			return;
		}
	}

	bool bLog = _args.value(ARG_NAME_SCALE).value<PVCore::PVEnumType>().get_sel().compare("Log") == 0;

	out.get_selection() = in.get_selection();

	/* 1st round: we calculate all the frequencies */
	typedef QHash<QString, PVRow> lines_hash_t;
	lines_hash_t lines_hash;
	size_t nthreads = 0;
	lines_hash_t** lines_hash_array = nullptr;
	QString** keys = nullptr;
#pragma omp parallel
	{
	nthreads = omp_get_num_threads();
#pragma omp master
	{
		lines_hash_array = new lines_hash_t*[nthreads];
		keys = new QString*[nthreads];
		for (size_t ith=0; ith<nthreads; ith++) {
			lines_hash_array[ith] = new lines_hash_t();
			keys[ith] = new QString();
		}
	}
#pragma omp barrier
	size_t th_index = omp_get_thread_num();
	lines_hash_t& lines_hash_th = *(lines_hash_array[th_index]);
	QString& key = *(keys[th_index]);
#pragma omp for
	for (counter = 0; counter < nb_lines; counter++) {
		if (!should_cancel()) {
			if (!_view->get_line_state_in_pre_filter_layer(counter)) {
				continue;
			}
			PVRush::PVNraw::const_nraw_table_line nrawvalues = nraw.get_row(counter);
			key = PVRush::PVUtils::generate_key_from_axes_values(axes, nrawvalues);

			lines_hash_th.insert(key, lines_hash_th[key]+1);
		}
	}
	}

	if (!should_cancel()) {
		for (size_t ith = 0; ith < nthreads; ith++) {
			QHashIterator<QString, PVRow> i(*(lines_hash_array[ith]));
			while (i.hasNext()) {
				i.next();
				lines_hash.insert(i.key(), i.value() + lines_hash[i.key()]);
			}
		}
		QHashIterator<QString, PVRow> i(lines_hash);
		while (i.hasNext()) {
			i.next();
			highest_frequency = picviz_max(highest_frequency, i.value());
		}
	}

	for (size_t ith=0; ith<nthreads; ith++) {
		delete lines_hash_array[ith];
	}
	delete [] lines_hash_array;

	if (should_cancel()) {
		if (&in != &out) {
			out = in;
		}
		return;
	}


	/* 2nd round: we get the color from the ratio compared with the key and the frequency */
	const lines_hash_t& const_lines_hash = lines_hash;
#pragma omp parallel
	{
	QString& key = *(keys[omp_get_thread_num()]);
#pragma omp for
	for (counter = 0; counter < nb_lines; counter++) {
		if (!should_cancel()) {
			if (_view->get_line_state_in_pre_filter_layer(counter)) {
				PVRush::PVNraw::const_nraw_table_line nrawvalues = nraw.get_row(counter);
				key = PVRush::PVUtils::generate_key_from_axes_values(axes, nrawvalues);

				PVRow count_frequency = const_lines_hash[key];

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
	}

	for (size_t ith=0; ith<nthreads; ith++) {
		delete keys[ith];
	}
	delete [] keys;

	if (should_cancel()) {
		if (&in != &out) {
			out = in;
		}
		return;
	}

	BENCH_END(heatline, "heatline", 1, 1, sizeof(PVRow), nb_lines);
}

QList<PVCore::PVArgumentKey> Picviz::PVLayerFilterHeatlineBase::get_args_keys_for_preset() const
{
	QList<PVCore::PVArgumentKey> keys = get_default_args().keys();
	keys.removeAll(ARG_NAME_AXES);
	return keys;
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
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(PVCore::PVColorGradientDualSliderType());
	return args;
}

void Picviz::PVLayerFilterHeatlineSel::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	PVCore::PVColorGradientDualSliderType ratios = _args[ARG_NAME_COLORS].value<PVCore::PVColorGradientDualSliderType>();

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
	args[PVCore::PVArgumentKey(ARG_NAME_COLORS, ARG_DESC_COLORS)].setValue(PVCore::PVColorGradientDualSliderType());
	return args;
}

void Picviz::PVLayerFilterHeatlineSelAndCol::post(PVLayer& /*in*/, PVLayer& out, float ratio, PVRow line_id)
{
	// Colorize
	PVCore::PVColor color;
	QColor qcolor;
	get_args_for_preset().keys();
	qcolor.setHsvF((1.0 - ratio)/3.0, 1.0, 1.0);
	color.fromQColor(qcolor);

	out.get_lines_properties().line_set_rgb_from_color(line_id, color);

	// Select
	PVCore::PVColorGradientDualSliderType ratios = _args.value(ARG_NAME_COLORS).value<PVCore::PVColorGradientDualSliderType>();

	const float *v = ratios.get_positions();

	float fmin = v[0];
	float fmax = v[1];

	if ((ratio > fmax) || (ratio < fmin)) {
		out.get_selection().set_line(line_id, 0);
	}
}

IMPL_FILTER(Picviz::PVLayerFilterHeatlineSelAndCol)
