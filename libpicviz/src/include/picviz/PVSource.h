/**
 * \file PVSource.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVSOURCE_H
#define PICVIZ_PVSOURCE_H

#include <QString>
#include <QList>
#include <QStringList>
#include <QVector>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVFile.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVAxesCombination.h>

#include <pvkernel/rush/PVExtractor.h> 
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVSourceDescription.h>

#include <picviz/PVAxisComputation.h>
#include <picviz/PVScene.h>
#include <picviz/PVRoot.h>
#include <picviz/PVSource_types.h>

namespace Picviz {

/**
 * \class PVSource
 */
typedef typename PVCore::PVDataTreeObject<PVScene, PVMapped> data_tree_source_t;
class LibPicvizDecl PVSource: public data_tree_source_t
{
	friend class PVCore::PVSerializeObject;
	friend class PVRoot;
	friend class PVScene;
	friend class PVView;
	friend class PVPlotted;
	friend class PVCore::PVDataTreeAutoShared<PVSource>;
public:
	//typedef PVSource_p p_type;
	typedef children_t list_mapped_t;

protected:
	PVSource(PVRush::PVInputType::list_inputs_desc const& inputs, PVRush::PVSourceCreator_p sc, PVRush::PVFormat format);
	PVSource();
	PVSource(const PVSource& org);

public:
	~PVSource();

public:
	/* Functions */
	PVCol get_column_count() const;

	bool has_nraw_folder() const { return _nraw_folder.isNull() == false; }

	PVSource_sp clone_with_no_process();

	PVRush::PVNraw& get_rushnraw();
	const PVRush::PVNraw& get_rushnraw() const;

	QString get_value(PVRow row, PVCol col) const;
	inline PVCore::PVUnicodeString get_data_unistr_raw(PVRow row, PVCol column) const { return nraw->at_unistr(row, column); }

	PVRow get_row_count() const;

	PVRush::PVExtractor& get_extractor();
	PVRush::PVControllerJob_p extract();
	PVRush::PVControllerJob_p extract_from_agg_nlines(chunk_index start, chunk_index nlines);
	void wait_extract_end(PVRush::PVControllerJob_p job);

	bool load_from_disk();

	PVRush::PVInputType_p get_input_type() const;

	inline PVAxesCombination& get_axes_combination() { return _axes_combination; }
	inline PVAxesCombination const& get_axes_combination() const { return _axes_combination; }

	void create_default_view();

	QStringList const& get_invalid_elts() const { return _inv_elts; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	void process_from_source();

	PVRush::PVSourceCreator_p get_source_creator() const { return _src_plugin; }
	QString get_name() const { return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs); }
	QString get_format_name() const { return _extractor.get_format().get_format_name(); }
	QString get_window_name() const;
	QString get_tooltip() const;

	PVView* last_active_view() const { return _last_active_view; }

	PVView* current_view();
	PVView const* current_view() const;

	PVRush::PVFormat& get_format() { return _extractor.get_format(); }
	PVRush::PVFormat const& get_format() const { return _extractor.get_format(); }
	void set_format(PVRush::PVFormat const& format);

	void set_invalid_elts_mode(bool restore_inv_elts);

	void add_column(PVAxisComputation_f f_axis, Picviz::PVAxis const& axis);

	virtual QString get_serialize_description() const { return "Source: " + get_name(); }

	static PVSource_p create_source_from_description(PVScene_p scene_p, const PVRush::PVSourceDescription& descr)
	{
		PVSource_p src_p(
			scene_p,
			descr.get_inputs(),
			descr.get_source_creator(),
			descr.get_format()
		);

		return src_p;
	}

	PVRush::PVSourceDescription::shared_pointer create_description()
	{
		PVRush::PVSourceDescription::shared_pointer descr_p(
			new PVRush::PVSourceDescription(
				get_inputs(),
				get_source_creator(),
				get_format()
			)
		);

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
	void set_section_clicked(PVCol col, size_t pos) { _section_clicked.first = col; _section_clicked.second = pos; }
	const std::pair<size_t, size_t>& section_clicked() const { return _section_clicked; }

private:
	void add_column(Picviz::PVAxis const& axis);
	void set_views_consistent(bool cons);
	void set_mapping_function_in_extractor();

protected:
	virtual void set_parent_from_ptr(PVScene* parent);
	virtual QString get_children_description() const { return "Mapped(s)"; }
	virtual QString get_children_serialize_name() const { return "mapped"; }

	void add_view(PVView_sp view);
	void set_views_id();

	inline void set_last_active_view(Picviz::PVView* view) { _last_active_view = view; }

protected:
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);
	PVSERIALIZEOBJECT_SPLIT

private:
	PVRush::PVRawSourceBase_p create_extractor_source(QString type, QString filename, PVRush::PVFormat const& format);
	void files_append_noextract();
	void init();
	void extract_finished();

private:
	PVView* _last_active_view = nullptr;

	PVRush::PVExtractor _extractor;
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	PVRush::PVInputType::list_inputs _inputs;

	PVRush::PVSourceCreator_p _src_plugin;
	PVRush::PVNraw *nraw;
	bool _restore_inv_elts;
	QStringList _inv_elts;

	PVAxesCombination _axes_combination;

	int _axis_hovered_id = -1;
	PVCol _axis_clicked_id;
	int _section_hovered_id = -1;
	std::pair<size_t, size_t> _section_clicked;

	QString _nraw_folder;
};

}

#endif	/* PICVIZ_PVSOURCE_H */
