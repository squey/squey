/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCE_H
#define INENDI_PVSOURCE_H

#include <QString>
#include <QList>
#include <QStringList>
#include <QVector>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVAxesCombination.h>

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVSourceDescription.h>

#include <inendi/PVAxisComputation.h>
#include <inendi/PVMapped.h>
#include <inendi/PVSource_types.h>

namespace Inendi
{

/**
 * \class PVSource
 */
typedef typename PVCore::PVDataTreeObject<PVScene, PVMapped> data_tree_source_t;
class PVSource : public data_tree_source_t
{
	friend class PVCore::PVSerializeObject;
	friend class PVRoot;
	friend class PVScene;
	friend class PVView;
	friend class PVPlotted;

  public:
	typedef children_t list_mapped_t;

  public:
	PVSource(Inendi::PVScene* scene,
	         PVRush::PVInputType::list_inputs_desc const& inputs,
	         PVRush::PVSourceCreator_p sc,
	         PVRush::PVFormat format);
	PVSource(PVScene* scene, const PVRush::PVSourceDescription& descr): PVSource(scene, descr.get_inputs(), descr.get_source_creator(), descr.get_format()) { }
	~PVSource();

  public:
	/* Functions */
	PVCol get_column_count() const;

	bool has_nraw_folder() const { return _nraw_folder.isNull() == false; }

	PVSource_sp clone_with_no_process();

	PVRush::PVNraw& get_rushnraw();
	const PVRush::PVNraw& get_rushnraw() const;

	std::string get_value(PVRow row, PVCol col) const;

	/**
	 * Return the number of row in the datastorage.
	 */
	PVRow get_row_count() const;

	/**
	 * Return number of correctly splitted row in the datastorage.
	 */
	PVRow get_valid_row_count() const { return get_row_count() - _inv_elts.size(); }

	PVRush::PVExtractor const& get_extractor() const { return _extractor; }

	/**
	 * This one is call by extractor widget after a source clone.
	 *
	 * @fixme: Should be remove so we can use the new one form new source.
	 */
	PVRush::PVExtractor& get_extractor() { return _extractor; }

	/**
	 * Start extraction of data for current source.
	 *
	 * @param line_count : Number of line to load
	 * @param skip_lines_count : Number of line to skip at the beginning of the file.
	 *
	 * @return : Pointer to the started job.
	 */
	PVRush::PVControllerJob_p extract(size_t skip_lines_count = 0, size_t line_count = 0);
	PVRush::PVControllerJob_p extract_from_agg_nlines(chunk_index start, chunk_index nlines);
	void wait_extract_end(PVRush::PVControllerJob_p job);

	bool load_from_disk();

	PVRush::PVInputType_p get_input_type() const;

	inline PVAxesCombination& get_axes_combination() { return _axes_combination; }
	inline PVAxesCombination const& get_axes_combination() const { return _axes_combination; }

	void create_default_view();

	std::map<size_t, std::string> const& get_invalid_evts() const { return _inv_elts; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	void process_from_source();

	PVRush::PVSourceCreator_p get_source_creator() const { return _src_plugin; }
	QString get_name() const
	{
		return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs);
	}
	QString get_format_name() const { return _extractor.get_format().get_format_name(); }
	QString get_window_name() const;
	QString get_tooltip() const;

	PVView* last_active_view() const { return _last_active_view; }

	PVView* current_view();
	PVView const* current_view() const;

	PVRush::PVFormat& get_format() { return _extractor.get_format(); }
	PVRush::PVFormat const& get_format() const { return _extractor.get_format(); }
	void set_format(PVRush::PVFormat const& format);

	void add_column(PVAxisComputation_f f_axis, Inendi::PVAxis const& axis);

	virtual QString get_serialize_description() const { return "Source: " + get_name(); }

	PVRush::PVSourceDescription::shared_pointer create_description()
	{
		PVRush::PVSourceDescription::shared_pointer descr_p(
		    new PVRush::PVSourceDescription(get_inputs(), get_source_creator(), get_format()));

		return descr_p;
	}

	size_t get_extraction_last_nlines() const { return _extractor.get_last_nlines(); }
	size_t get_extraction_last_start() const { return _extractor.get_last_start(); }

	// axis <-> section synchronisation
	void set_axis_hovered(PVCol col, bool entered) { _axis_hovered_id = entered ? col : -1; }
	int& axis_hovered() { return _axis_hovered_id; }
	const int& axis_hovered() const { return _axis_hovered_id; }
	void set_axis_clicked(PVCol col) { _axis_clicked_id = col; }
	const PVCol& axis_clicked() const { return _axis_clicked_id; }
	void set_section_hovered(PVCol col, bool entered) { _section_hovered_id = entered ? col : -1; }
	const int& section_hovered() const { return _section_hovered_id; }
	void set_section_clicked(PVCol col, size_t pos)
	{
		_section_clicked.first = col;
		_section_clicked.second = pos;
	}
	const std::pair<size_t, size_t>& section_clicked() const { return _section_clicked; }

  private:
	void add_column(Inendi::PVAxis const& axis);
	void set_views_consistent(bool cons);

  protected:
	virtual QString get_children_description() const { return "Mapped(s)"; }
	virtual QString get_children_serialize_name() const { return "mapped"; }

	void add_view(PVView* view);
	void set_views_id();

	inline void set_last_active_view(Inendi::PVView* view) { _last_active_view = view; }

  protected:
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

  private:
	PVRush::PVRawSourceBase_p
	create_extractor_source(QString type, QString filename, PVRush::PVFormat const& format);
	void files_append_noextract();
	void extract_finished();

  private:
	PVView* _last_active_view = nullptr;

	PVRush::PVExtractor _extractor; //!< Tool to extract data and generate NRaw.
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	PVRush::PVInputType::list_inputs _inputs;

	PVRush::PVSourceCreator_p _src_plugin;
	PVRush::PVNraw& _nraw;                    //!< Pointer to Nraw data (owned by extractor)
	std::map<size_t, std::string> _inv_elts; //!< List of invalid elements sorted by line number.

	PVAxesCombination _axes_combination;

	int _axis_hovered_id = -1;
	PVCol _axis_clicked_id;
	int _section_hovered_id = -1;
	std::pair<size_t, size_t> _section_clicked;

	QString _nraw_folder;
};
}

#endif /* INENDI_PVSOURCE_H */
