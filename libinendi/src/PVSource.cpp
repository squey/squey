/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/debug.h>

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include <inendi/general.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>
#include <inendi/PVRoot.h>

Inendi::PVSource::PVSource(Inendi::PVScene* scene,
                           PVRush::PVInputType::list_inputs const& inputs,
                           PVRush::PVSourceCreator_p sc,
                           PVRush::PVFormat format)
    : PVSource(scene,
               inputs,
               sc,
               format,
               0,
               PVCore::PVConfig::get()
                   .config()
                   .value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT)
                   .toInt())
{
}

Inendi::PVSource::PVSource(Inendi::PVScene* scene,
                           PVRush::PVInputType::list_inputs const& inputs,
                           PVRush::PVSourceCreator_p sc,
                           PVRush::PVFormat format,
                           size_t ext_start,
                           size_t ext_end)
    : data_tree_source_t(scene), _inputs(inputs), _src_plugin(sc), _nraw(_extractor.get_nraw())
{
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	// Set extractor default values
	_extractor.set_last_start(ext_start);
	_extractor.set_last_nlines(ext_end);

	int nchunks = pvconfig.value("pvkernel/number_living_chunks", 0).toInt();
	if (nchunks != 0) {
		_extractor.set_number_living_chunks(nchunks);
	}

	// Set format
	format.populate();
	set_format(format);

	// Set sources
	files_append_noextract();
}

Inendi::PVSource::~PVSource()
{
	PVRoot* root = get_parent<PVRoot>();
	if (root) {
		root->source_being_deleted(this);
	}
	remove_all_children();
	PVLOG_DEBUG("In PVSource destructor: %p\n", this);
}

Inendi::PVSource_sp Inendi::PVSource::clone_with_no_process()
{
	// FIXME : Should be remove
	Inendi::PVSource_sp src = get_parent()->emplace_add_child(_inputs, _src_plugin, get_format());
	src->emplace_add_child();
	return src;
}

Inendi::PVView* Inendi::PVSource::current_view()
{
	PVView* view = get_parent<PVRoot>()->current_view();
	if (view->get_parent<PVSource>() == this) {
		return view;
	}
	return nullptr;
}

Inendi::PVView const* Inendi::PVSource::current_view() const
{
	PVView const* view = get_parent<PVRoot>()->current_view();
	if (view->get_parent<PVSource>() == this) {
		return view;
	}
	return nullptr;
}

void Inendi::PVSource::files_append_noextract()
{
	for (int i = 0; i < _inputs.count(); i++) {
		PVRush::PVSourceCreator::source_p src =
		    _src_plugin->create_source_from_input(_inputs[i], _extractor.get_format());
		_extractor.add_source(src);
	}
}

PVRush::PVControllerJob_p Inendi::PVSource::extract(size_t skip_lines_count /*= 0*/,
                                                    size_t line_count /*= 0*/)
{
	// Set all views as non-consistent
	for (auto view_p : get_children<PVView>()) {
		view_p->set_consistent(false);
	}

	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(
	    skip_lines_count, line_count ? line_count : INENDI_LINES_MAX);

	return job;
}

PVRush::PVControllerJob_p Inendi::PVSource::extract_from_agg_nlines(chunk_index start,
                                                                    chunk_index nlines)
{
	// Set all views as non-consistent
	for (auto view_p : get_children<PVView>()) {
		view_p->set_consistent(false);
	}

	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(start, nlines);
	return job;
}

void Inendi::PVSource::wait_extract_end(PVRush::PVControllerJob_p job)
{
	job->wait_end();
	_inv_elts = job->get_invalid_evts();
	extract_finished();
}

bool Inendi::PVSource::load_from_disk()
{
	return _nraw.load_from_disk(_nraw_folder.toStdString());
}

void Inendi::PVSource::extract_finished()
{
	_extractor.get_agg().release_inputs();
}

void Inendi::PVSource::set_format(PVRush::PVFormat const& format)
{
	_extractor.set_format(format);
	_extractor.get_format().restore_invalid_evts(true);
	_axes_combination.set_from_format(_extractor.get_format());

	PVFilter::PVChunkFilterByElt* chk_flt = _extractor.get_format().create_tbb_filters();
	_extractor.set_chunk_filter(chk_flt);
}

PVRush::PVNraw& Inendi::PVSource::get_rushnraw()
{
	return _nraw;
}

const PVRush::PVNraw& Inendi::PVSource::get_rushnraw() const
{
	return _nraw;
}

PVRow Inendi::PVSource::get_row_count() const
{
	return _nraw.get_row_count();
}

PVCol Inendi::PVSource::get_column_count() const
{
	return get_format().get_axes().size();
}

std::string Inendi::PVSource::get_value(PVRow row, PVCol col) const
{
	return _nraw.at_string(row, col);
}

PVRush::PVInputType_p Inendi::PVSource::get_input_type() const
{
	assert(_src_plugin);
	return _src_plugin->supported_type_lib();
}

void Inendi::PVSource::create_default_view()
{
	if (get_children_count() == 0) {
		emplace_add_child();
	}
	for (PVMapped_p& m : get_children()) {
		PVPlotted_p def_plotted = m->emplace_add_child();

		PVView_p def_view = def_plotted->emplace_add_child();
		def_view->get_parent<PVRoot>()->select_view(*def_view);
		process_from_source();
	}
}

void Inendi::PVSource::process_from_source()
{
	for (auto mapped_p : get_children<PVMapped>()) {
		mapped_p->process_from_parent_source();
	}
}

void Inendi::PVSource::add_view(PVView* view)
{
	PVRoot* root = get_parent<PVRoot>();
	root->select_view(*view);
	view->set_view_id(root->get_new_view_id());
	view->set_color(root->get_new_view_color());
}

void Inendi::PVSource::add_column(PVAxis const& axis)
{
	PVCol new_col_idx = get_rushnraw().get_number_cols() - 1;
	PVMappingProperties map_prop(axis, new_col_idx);

	// Add that column to our children
	for (auto mapped : get_children<PVMapped>()) {
		mapped->add_column(map_prop);
		PVPlottingProperties plot_prop(mapped->get_mapping(), axis, new_col_idx);
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

void Inendi::PVSource::set_views_consistent(bool cons)
{
	for (auto view_p : get_children<PVView>()) {
		view_p->set_consistent(cons);
	}
}

void Inendi::PVSource::add_column(PVAxisComputation_f f_axis, PVAxis const& axis)
{
	set_views_consistent(false);
	if (f_axis(&get_rushnraw())) {
		add_column(axis);
	}
	set_views_consistent(true);
}

QString Inendi::PVSource::get_window_name() const
{
	const size_t line_start = get_extraction_last_start();
	const size_t line_end = line_start + get_row_count() - 1;
	return get_name() + QString(" / ") + get_format_name() +
	       QString("\n(%L1 -> %L2)").arg(line_start).arg(line_end);
}

QString Inendi::PVSource::get_tooltip() const
{
	const size_t line_start = get_extraction_last_start();
	const size_t line_end = line_start + get_row_count() - 1;

	QString format = QString("format: %1").arg(get_format_name());
	QString range = QString("range: %L1 - %L2").arg(line_start).arg(line_end);

	return format + "\n" + range;
}

void Inendi::PVSource::serialize_write(PVCore::PVSerializeObject& so)
{
	QString src_name = _src_plugin->registered_name();
	so.attribute("source-plugin", src_name);

	// Save the state of the extractor
	chunk_index start, nlines;
	start = _extractor.get_last_start();
	nlines = _extractor.get_last_nlines();
	so.attribute("index_start", start);
	so.attribute("nlines", nlines);

	QString nraw_path = QString::fromStdString(get_rushnraw().collection().rootdir());
	so.attribute("nraw_path", nraw_path, QString());

	// Save the format
	so.object("format", _extractor.get_format(), QObject::tr("Format"));

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	for (PVCore::PVSharedPtr<PVMapped> mapped : get_children()) {
		QString child_name = QString::number(idx);
		PVCore::PVSerializeObject_p new_obj =
		    list_obj->create_object(child_name, mapped->get_serialize_description(), false);
		mapped->serialize(*new_obj, so.get_version());
		new_obj->_bound_obj = mapped.get();
		new_obj->_bound_obj_type = typeid(PVMapped);
	}
}

void Inendi::PVSource::serialize_read(PVCore::PVSerializeObject& so,
                                      PVCore::PVSerializeArchive::version_t v)
{
	// Create the list of mapped
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	int idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
			PVMapped_p mapped = emplace_add_child();
			// FIXME : Mapping is created invalid then set
			new_obj->object(QString("mapping"), mapped->get_mapping(), QString(), false, nullptr,
			                false);
			mapped->serialize(*new_obj, so.get_version());
			new_obj->_bound_obj = mapped.get();
			new_obj->_bound_obj_type = typeid(PVMapped);
			idx++;
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
		return;
	}
}
