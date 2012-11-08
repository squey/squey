/**
 * \file PVSource.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
	PVLOG_INFO("In PVSource destructor: %p\n", this);
	_extractor.force_stop_controller();
	for (auto& m: get_children()) {
		PVMapped* pm = m.get();
		m.reset();
		PVLOG_INFO("Mapped %p use count: %u\n", pm, pm->weak_from_this().use_count());
	}
}

void Picviz::PVSource::init()
{
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

Picviz::PVSource_sp Picviz::PVSource::clone_with_no_process() const
{
	//Picviz::PVSource_p src(get_parent(), _inputs, _src_plugin, get_format());
	return Picviz::PVSource_sp();
}

Picviz::PVView* Picviz::PVSource::current_view()
{
	PVView* view = get_parent<PVScene>()->current_view();
	return view;
}

Picviz::PVView const* Picviz::PVSource::current_view() const
{
	return const_cast<PVView const*>(const_cast<PVSource*>(this)->current_view());
}

void Picviz::PVSource::set_parent_from_ptr(PVScene* parent)
{
	data_tree_source_t::set_parent_from_ptr(parent);

	if (parent) {
		parent->get_parent<PVRoot>()->set_views_id();
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

	set_mapping_function_in_extractor();

	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines_last_param();
	return job;
}

PVRush::PVControllerJob_p Picviz::PVSource::extract_from_agg_nlines(chunk_index start, chunk_index nlines)
{
	// Set all views as non-consistent
	decltype(get_children<PVView>())::iterator it_view;
	for (auto view_p : get_children<PVView>()) {
		view_p->set_consistent(false);
	}

	set_mapping_function_in_extractor();

	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(start, nlines);
	return job;
}

void Picviz::PVSource::set_mapping_function_in_extractor()
{
	PVRush::PVNrawOutput::list_chunk_functions& funcs = _extractor.chunk_functions();
	PVFilter::PVPureMappingProcessing::list_pure_mapping_t& m_funcs = _extractor.pure_mapping_functions();
	funcs.clear();
	m_funcs.clear();

	children_t const& mappeds = get_children();
	if (mappeds.size() == 0) {
		return;
	}

	for (auto m: mappeds) {
		funcs.emplace_back(boost::bind<void>(&PVMapped::process_rush_pipeline_chunk, m.get(), _1, _2));
		m->init_process_from_rush_pipeline();
	}

	// AG: *HACK*: only the first mapping get a placeholder for its pure mapped values...
	Picviz::PVMapped& first_child = *mappeds.at(0);
	first_child.init_pure_mapping_functions(m_funcs);
}

void Picviz::PVSource::wait_extract_end(PVRush::PVControllerJob_p job)
{
	job->wait_end();
	_inv_elts = job->get_invalid_elts();
	extract_finished();
}

void Picviz::PVSource::extract_finished()
{
	// Finish mapping process. That will set all mapping as valid!
	for (auto mapped_p : get_children<PVMapped>()) {
		mapped_p->finish_process_from_rush_pipeline();
	}

	// Reset all views and process the current one
	for (auto view_p : get_children<PVView>()) {
		view_p->reset_layers();
	}
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

PVRush::PVNraw& Picviz::PVSource::get_rushnraw()
{
	return *nraw;
}

const PVRush::PVNraw& Picviz::PVSource::get_rushnraw() const
{
	return *nraw;
}

PVRow Picviz::PVSource::get_row_count() const
{
	return nraw->get_number_rows();
}

PVCol Picviz::PVSource::get_column_count() const
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
	if (get_children_count() == 0) {
		PVMapped_p def_mapped(shared_from_this());
	}
	for (PVMapped_p& m: get_children()) {
		PVPlotted_p def_plotted(m);
		PVView_p def_view(def_plotted);
		def_view->get_parent<PVRoot>()->select_view(*def_view);
		process_from_source();
	}
}

void Picviz::PVSource::process_from_source()
{
	for (auto mapped_p : get_children<PVMapped>()) {
		mapped_p->process_from_parent_source();
	}
}

void Picviz::PVSource::add_view(PVView_sp view)
{
	//if (!current_view()) {
		get_parent<PVRoot>()->select_view(*view);
	//}
	PVRoot* root = get_parent<PVRoot>();
	if (root) {
		view->set_view_id(root->get_new_view_id());
		view->set_color(root->get_new_view_color());
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
	process_from_source();
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
		//get_parent()->add_source(shared_from_this());

		// "Append" the files to the extractor
		files_append_noextract();
	}

	data_tree_source_t::serialize_read(so, v);
}
