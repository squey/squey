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
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

Picviz::PVSource::PVSource(PVRush::PVInputType::list_inputs const& inputs, PVRush::PVSourceCreator_p sc, PVRush::PVFormat format):
	data_tree_source_t()
{
	init();

	// Set format
	format.populate();
	set_format(format);

	// Set sources
	_inputs = inputs;
	_src_plugin = sc;
	files_append_noextract();
}

Picviz::PVSource::PVSource():
	data_tree_source_t()
{
	init();
}

Picviz::PVSource::PVSource(const PVSource& org):
	data_tree_source_t()
{
	set_parent(const_cast<PVScene*>(org.get_parent()));
	init();
}

Picviz::PVSource::~PVSource()
{
	PVLOG_DEBUG("In PVSource destructor\n");
	_extractor.force_stop_controller();
}

void Picviz::PVSource::init()
{
	_current_view.reset();
	nraw = &(_extractor.get_nraw());
	// Set extractor default values
	_extractor.set_last_start(0);
	_extractor.set_last_nlines(pvconfig.value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT).toInt());

	int nchunks = pvconfig.value("pvkernel/number_living_chunks", 0).toInt();
	if (nchunks != 0) {
		_extractor.set_number_living_chunks(nchunks);
	}

	_restore_inv_elts = false;

	// Launch the controller thread
	_extractor.start_controller();
}

void Picviz::PVSource::set_parent_from_ptr(PVScene* parent)
{
	data_tree_source_t::set_parent_from_ptr(parent);

	if (parent) {
		parent->set_views_id();
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
	// Set all views as non-consistent
	for (auto view_p : get_children<PVView>()) {
		view_p->set_consistent(false);
	}

	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines_last_param();
	return job;
}

PVRush::PVControllerJob_p Picviz::PVSource::extract_from_agg_nlines(chunk_index start, chunk_index nlines)
{
	// Set all views as non-consistent
	list_views_t::iterator it_view;
	for (auto view_p : get_children<PVView>()) {
		view_p->set_consistent(false);
	}

	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(start, nlines);
	return job;
}

void Picviz::PVSource::wait_extract_end(PVRush::PVControllerJob_p job)
{
	job->wait_end();
	_inv_elts = job->get_invalid_elts();
	extract_finished();
}

void Picviz::PVSource::select_view(PVView_sp view)
{
	 assert(get_children<PVView>().contains(view));
	 _current_view = view;
}

void Picviz::PVSource::extract_finished()
{
	// Set all mapped children as invalid
	for (auto mapped_p : get_children<PVMapped>()) {
		mapped_p->invalidate_all();
	}

	// Reset all views and process the current one
	/*for (auto view_p : get_children<PVView>()) {
		view_p->reset_layers();
	}*/
}

void Picviz::PVSource::set_format(PVRush::PVFormat const& format)
{
	_extractor.set_format(format);
	if (_restore_inv_elts) {
		_extractor.get_format().restore_invalid_elts(true);
		_extractor.dump_inv_elts(true);
	}
	else {
		_extractor.get_format().restore_invalid_elts(false);
		_extractor.dump_inv_elts(false);
	}
	_axes_combination.set_from_format(_extractor.get_format());

	PVFilter::PVChunkFilter_f chk_flt = _extractor.get_format().create_tbb_filters();
	_extractor.set_chunk_filter(chk_flt);
}

PVRush::PVNraw::nraw_table& Picviz::PVSource::get_qtnraw()
{
	return nraw->get_table();
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
	return nraw->get_trans_table();
}

void Picviz::PVSource::clear_trans_nraw()
{
	nraw->free_trans_nraw();
}

const PVRush::PVNraw::nraw_table& Picviz::PVSource::get_qtnraw() const
{
	return nraw->get_table();
}

PVRow Picviz::PVSource::get_row_count()
{
	return nraw->get_number_rows();
}

PVCol Picviz::PVSource::get_column_count()
{
	return get_format().get_axes().size();
}

QString Picviz::PVSource::get_value(PVRow row, PVCol col) const
{
	return nraw->at(row,col);
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

void Picviz::PVSource::create_default_view()
{
	//new PVPlotted(new PVMapped(this));
	PVMapped_p def_mapped;
	def_mapped->set_parent(this);
	def_mapped->process_from_parent_source(false);
	PVPlotted_p def_plotted;
	def_plotted->set_parent(def_mapped);
	def_plotted->process_from_parent_mapped(false);
}

void Picviz::PVSource::process_from_source(bool keep_views_info)
{
	for (auto mapped_p : get_children<PVMapped>()) {
		mapped_p->process_from_source(this, keep_views_info);
	}
}

void Picviz::PVSource::add_view(PVView_sp view)
{
	auto views_p = get_children<PVView>();
	if (!_current_view) {
		_current_view = view;
	}
	if (!views_p.contains(view)) {
		PVScene* scene = get_parent();
		if (scene) {
			view->set_view_id(scene->get_new_view_id());
		}
	}
}

void Picviz::PVSource::add_column(PVAxis const& axis)
{
	PVCol new_col_idx = get_rushnraw().get_number_cols()-1;
	PVMappingProperties map_prop(axis, new_col_idx);

	// Add that column to our children
	for (auto mapped : get_children<PVMapped>()) {
		mapped->add_column(map_prop);
		PVPlottingProperties plot_prop(*mapped->get_mapping(), axis, new_col_idx);
		for (auto plotted : mapped->get_children<PVPlotted>()) {
			plotted->add_column(plot_prop);
		}
	}
	for (auto view_p : get_children<PVView>()) {
		view_p->add_column(axis);
	}

	// Reprocess from source
	process_from_source(true);
}

void Picviz::PVSource::set_views_consistent(bool cons)
{
	for (auto view_p : get_children<PVView>()) {
		view_p->set_consistent(cons);
	}
}

void Picviz::PVSource::add_column(PVAxisComputation_f f_axis, PVAxis const& axis)
{
	set_views_consistent(false);
	if (f_axis(&get_rushnraw())) {
		add_column(axis);
	}
	set_views_consistent(true);
}

void Picviz::PVSource::set_invalid_elts_mode(bool restore_inv_elts)
{
	_restore_inv_elts = restore_inv_elts;
	PVRush::PVFormat format = _extractor.get_format();
	set_format(format);
}

void Picviz::PVSource::serialize_write(PVCore::PVSerializeObject& so)
{
	data_tree_source_t::serialize_write(so);

	PVRush::PVInputType_p in_t = get_input_type();
	assert(in_t);
	PVCore::PVSerializeObject_p so_inputs = get_parent<PVScene>()->get_so_inputs(*this);
	if (so_inputs) {
		// The inputs have been serialized by our parents, so just make references to them
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
	so.object("format", _extractor.get_format(), QObject::tr("Format"));
}

void Picviz::PVSource::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v)
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
	assert(get_parent<PVScene>());
	PVCore::PVSerializeObject_p so_inputs = get_parent<PVScene>()->get_so_inputs(*this);
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
	if (!so.has_repairable_errors()) {
		set_format(format);
		get_parent()->add_source(shared_from_this());

		// "Append" the files to the extractor
		files_append_noextract();
	}

	data_tree_source_t::serialize_read(so, v);
}
