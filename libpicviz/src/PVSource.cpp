//! \file PVSource.cpp
//! $Id: PVSource.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/debug.h>

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
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

Picviz::PVSource::PVSource(PVRush::PVInputType::list_inputs const& inputs, PVRush::PVSourceCreator_p sc, PVRush::PVFormat format)
{
	init();

	// Set format
	format.populate();
	set_format(format);

	// Set sources
	_inputs = inputs;
	_src_plugin = sc;
	files_append_noextract();

	set_parent(NULL);
}

Picviz::PVSource::PVSource()
{
	init();
}

Picviz::PVSource::PVSource(const PVSource& org):
	boost::enable_shared_from_this<PVSource>()
{
	init();
	root = org.root;
	tparent = org.tparent;
}

Picviz::PVSource::~PVSource()
{
	PVLOG_INFO("In PVSource destructor\n");
	_extractor.force_stop_controller();
}

void Picviz::PVSource::init()
{
	nraw = &(_extractor.get_nraw());
	// Set extractor default values
	_extractor.set_last_start(0);
	_extractor.set_last_nlines(pvconfig.value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT).toInt());

	// Launch the controller thread
	_extractor.start_controller();
}

void Picviz::PVSource::set_parent(PVScene* parent)
{
	if (parent) {
		tparent = parent;
		root = parent->get_root();
	}
	else {
		tparent = NULL;
		root = NULL;
	}
}

void Picviz::PVSource::files_append_noextract()
{
	for (int i = 0; i < _inputs.count(); i++) {
		PVRush::PVSourceCreator::source_p src = _src_plugin->create_source_from_input(_inputs[i], _extractor.get_format());
		_extractor.add_source(src);
	}
}

PVRush::PVControllerJob_p Picviz::PVSource::extract()
{
	return _extractor.process_from_agg_nlines_last_param();
}

void Picviz::PVSource::set_format(PVRush::PVFormat const& format)
{
	_extractor.set_format(format);
	_axes_combination.set_from_format(_extractor.get_format());

	PVFilter::PVChunkFilter_f chk_flt = _extractor.get_format().create_tbb_filters();
	_extractor.set_chunk_filter(chk_flt);
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

QString const& Picviz::PVSource::get_value(PVRow row, PVCol col) const
{
	return nraw->table.at(row)[col];
}

PVRush::PVExtractor& Picviz::PVSource::get_extractor()
{
	return _extractor;
}

PVRush::PVInputType_p Picviz::PVSource::get_input_type() const
{
	assert(_src_plugin);
	return _src_plugin->supported_type_lib();
}

void Picviz::PVSource::process_from_source(bool keep_layers)
{
	// Init all views from this source
	list_views_t::iterator it;
	for (it = _views.begin(); it != _views.end(); it++) {
		(*it)->init_from_source(shared_from_this(), keep_layers);
		(*it)->process_from_layer_stack();
	}
}


void Picviz::PVSource::add_view(PVView_p view)
{
	view->init_from_source(shared_from_this(), false);
	view->process_from_layer_stack();
	_views.push_back(view);
}

bool Picviz::PVSource::del_view(const PVView* view)
{
	list_views_t::iterator it;
	for (it = _views.begin(); it != _views.end(); it++) {
		if (it->get() == view) {
			_views.erase(it);
			return true;
		}
	}
	return false;
}

void Picviz::PVSource::serialize_write(PVCore::PVSerializeObject& so)
{
	PVRush::PVInputType_p in_t = get_input_type();
	assert(in_t);
	PVCore::PVSerializeObject_p so_inputs = tparent->get_so_inputs(*this);
	if (so_inputs) {
		// The inputs have bee nserialized by our parents, so just make references to them
		in_t->serialize_inputs_ref(so, "inputs", _inputs, so_inputs);
	}
	else {
		// Serialize the inputs
		in_t->serialize_inputs(so, "inputs", _inputs);
	}
	QString src_name = _src_plugin->registered_name();
	so.attribute("source-plugin", src_name);

	// Save the state of the extractor
	chunk_index start, nlines;
	start = _extractor.get_last_start();
	nlines = _extractor.get_last_nlines();
	so.attribute("index_start", start);
	so.attribute("nlines", nlines);

	// Save the format
	so.object("format", _extractor.get_format());

	// Save the views
	so.list("views", _views);
}

void Picviz::PVSource::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	QString src_name;
	so.attribute("source-plugin", src_name);
	PVRush::PVSourceCreator_p sc_lib = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(src_name);
	if (!sc_lib) {
		return;
	}
	// Get the source plugin
	_src_plugin = sc_lib->clone<PVRush::PVSourceCreator>();

	// Get the inputs
	PVCore::PVSerializeObject_p so_inputs = tparent->get_so_inputs(*this);
	if (so_inputs) {
		// The inputs have been serialized by our parents, so just make references to them
		get_input_type()->serialize_inputs_ref(so, "inputs", _inputs, so_inputs);
	}
	else {
		// Serialize the inputs
		get_input_type()->serialize_inputs(so, "inputs", _inputs);
	}

	// Get the state of the extractor
	chunk_index start, nlines;
	so.attribute("index_start", start);
	so.attribute("nlines", nlines);
	_extractor.set_last_start(start);
	_extractor.set_last_nlines(nlines);

	// Get the format
	PVRush::PVFormat format;
	so.object("format", format);
	set_format(format);

	// "Append" the files to the extractor
	files_append_noextract();

	// Load the views
	so.list("views", _views);
}
