//! \file PVSource.cpp
//! $Id: PVSource.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVFieldSplitterUTF16Char.h>
#include <pvkernel/filter/PVFieldsMappingFilter.h>

#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVInputPcap.h>
#include <pvkernel/rush/PVChunkAlignUTF16Char.h>
#include <pvkernel/rush/PVChunkTransformUTF16.h>
#include <pvkernel/rush/PVRawSource.h>
#include <pvkernel/rush/PVUnicodeSource.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVControllerJob.h>

#include <picviz/general.h>
#include <pvkernel/core/debug.h>

#include <QRegExp>

Picviz::PVSource::PVSource(PVScene_p parent)
{
	tparent = parent;
	root = parent->root;
	nraw = &(_extractor.get_nraw());

	// Launch the controller thread
	_extractor.start_controller();
}

Picviz::PVSource::~PVSource()
{
	PVLOG_INFO("In PVSource destructor\n");
	_extractor.force_stop_controller();
}

void Picviz::PVSource::files_append_noextract(PVRush::PVFormat const& format, PVRush::PVSourceCreator_p sc, PVRush::PVInputType::list_inputs inputs)
{
	set_format(format);
	for (int i = 0; i < inputs.count(); i++) {
		PVRush::PVSourceCreator::source_p src = sc->create_source_from_input(inputs[i]);
		_extractor.add_source(src);
	}
}

void Picviz::PVSource::set_format(PVRush::PVFormat const& format)
{
	PVRush::PVFormat *format_nraw = new PVRush::PVFormat(format);
	_extractor.get_nraw().format.reset(format_nraw);
	format_nraw->populate();
	axes_combination.set_from_format(*format_nraw);

	PVFilter::PVChunkFilter_f chk_flt = format_nraw->create_tbb_filters();
	_extractor.set_chunk_filter(chk_flt);
}

PVRush::PVControllerJob_p Picviz::PVSource::files_append(PVRush::PVFormat const& format, PVRush::PVSourceCreator_p sc, PVRush::PVInputType::list_inputs inputs)
{
	files_append_noextract(format, sc, inputs);
	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(0, pvconfig.value("pvkernel/rush/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT).toInt());

	return job;
}

PVRush::PVNraw::nraw_table& Picviz::PVSource::get_qtnraw()
{
	return nraw->table;
}

PVRush::PVNraw& Picviz::PVSource::get_rushnraw()
{
	return *nraw;
}

const PVRush::PVNraw& Picviz::PVSource::get_rushnraw() const
{
	return *nraw;
}

PVRush::PVNraw::nraw_trans_table const& Picviz::PVSource::get_trans_nraw() const
{
	return nraw->trans_table;
}

void Picviz::PVSource::clear_trans_nraw()
{
	nraw->free_trans_nraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVSource::get_qtnraw() const
{
	return nraw->table;
}

PVRow Picviz::PVSource::get_row_count()
{
	return nraw->table.size();
}

PVCol Picviz::PVSource::get_column_count()
{
	return nraw->table.at(0).size();
}

QString Picviz::PVSource::get_value(PVRow row, PVCol col)
{
	return nraw->table.at(row)[col];
}

void Picviz::PVSource::set_limits(PVRow min, PVRow max)
{
	limit_min = min;
	limit_max = max;
}

PVRush::PVExtractor& Picviz::PVSource::get_extractor()
{
	return _extractor;
}
