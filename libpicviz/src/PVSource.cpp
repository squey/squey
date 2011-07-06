//! \file PVSource.cpp
//! $Id: PVSource.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>

#include <pvfilter/PVChunkFilterByElt.h>
#include <pvfilter/PVElementFilterByFields.h>
#include <pvfilter/PVFieldSplitterUTF16Char.h>
#include <pvfilter/PVFieldsMappingFilter.h>

#include <pvrush/PVInputFile.h>
#include <pvrush/PVInputPcap.h>
#include <pvrush/PVChunkAlignUTF16Char.h>
#include <pvrush/PVChunkTransformUTF16.h>
#include <pvrush/PVRawSource.h>
#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVNrawOutput.h>
#include <pvrush/PVControllerJob.h>

#include <picviz/general.h>
#include <pvcore/debug.h>

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

PVRush::PVControllerJob_p Picviz::PVSource::files_append(PVRush::PVFormat const& format, PVRush::PVSourceCreator_p sc, PVRush::PVInputType::list_inputs inputs)
{
	// FIXME: the format should be in the PVNraw
	PVRush::PVFormat *format_nraw = new PVRush::PVFormat(format);
	_extractor.get_nraw().format = format_nraw;
	format_nraw->populate();
	axes_combination.set_from_format(format_nraw);

	for (int i = 0; i < inputs.count(); i++) {
		PVRush::PVSourceCreator::source_p src = sc->create_source_from_input(inputs[i]);
		_extractor.add_source(src);
	}

	PVFilter::PVChunkFilter_f chk_flt = format_nraw->create_tbb_filters();
	_extractor.set_chunk_filter(chk_flt);

	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(0, PVEXTRACT_NUMBER_LINES_FIRST);

	return job;
}

PVRush::PVNraw::nraw_table& Picviz::PVSource::get_qtnraw()
{
	return nraw->table;
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
