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
	_inputs = inputs;
	_src_plugin = sc;
	PVRush::PVFormat format_ = format;
	format_.populate();
	for (int i = 0; i < inputs.count(); i++) {
		PVRush::PVSourceCreator::source_p src = sc->create_source_from_input(inputs[i], format_);
		_extractor.add_source(src);
	}
	set_format(format_);
}

PVRush::PVControllerJob_p Picviz::PVSource::reextract()
{
	return _extractor.process_from_agg_nlines_last_param();
}

void Picviz::PVSource::set_format(PVRush::PVFormat const& format)
{
	_extractor.set_format(format);
	axes_combination.set_from_format(_extractor.get_format());

	PVFilter::PVChunkFilter_f chk_flt = _extractor.get_format().create_tbb_filters();
	_extractor.set_chunk_filter(chk_flt);
}

PVRush::PVControllerJob_p Picviz::PVSource::files_append(PVRush::PVFormat const& format, PVRush::PVSourceCreator_p sc, PVRush::PVInputType::list_inputs inputs)
{
	files_append_noextract(format, sc, inputs);
	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(0, pvconfig.value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT).toInt());

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

PVRush::PVInputType_p Picviz::PVSource::get_input_type()
{
	assert(_src_plugin);
	QString in_name = _src_plugin->supported_type();
	PVRush::PVInputType_p lib = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(in_name);
	assert(lib);
	return lib->clone<PVRush::PVInputType>();
}

void Picviz::PVSource::serialize_write(PVCore::PVSerializeObject& so)
{
	PVRush::PVInputType_p in_t = get_input_type();
	in_t->serialize_inputs(so, "inputs", _inputs);
	//so.object("format", _extractor.get_format());
	QString src_name = _src_plugin->registered_name();
	so.attribute("source-plugin", src_name);
	chunk_index start, nlines;
	start = _extractor.get_last_start();
	nlines = _extractor.get_last_nlines();
	so.attribute("index_start", start);
	so.attribute("nlines", nlines);
}

void Picviz::PVSource::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	PVRush::PVFormat format;
	//so.object("format", format);
	QString src_name;
	so.attribute("source-plugin", src_name);
	PVRush::PVSourceCreator_p sc_lib = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(src_name);
	if (!sc_lib) {
		return;
	}
	_src_plugin = sc_lib->clone<PVRush::PVSourceCreator>();
	get_input_type()->serialize_inputs(so, "inputs", _inputs);
	files_append_noextract(format, _src_plugin, _inputs);
	chunk_index start, nlines;
	so.attribute("index_start", start);
	so.attribute("nlines", nlines);
	_extractor.set_last_start(start);
	_extractor.set_last_nlines(nlines);
}
