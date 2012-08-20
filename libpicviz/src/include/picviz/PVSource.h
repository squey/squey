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

#include <picviz/PVAxisComputation.h>
#include <picviz/PVScene.h>
#include <picviz/PVRoot.h>
#include <picviz/PVSource_types.h>

#include <boost/enable_shared_from_this.hpp>

namespace Picviz {

/**
 * \class PVSource
 */
typedef typename PVCore::PVDataTreeObject<PVScene, PVMapped> data_tree_source_t;
class LibPicvizDecl PVSource: public data_tree_source_t
{
	friend class PVCore::PVSerializeObject;
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
	PVCol get_column_count();

	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;

	PVRush::PVNraw& get_rushnraw();
	const PVRush::PVNraw& get_rushnraw() const;

	const PVRush::PVNraw::nraw_trans_table& get_trans_nraw() const;
	void clear_trans_nraw();

	QString get_value(PVRow row, PVCol col) const;
	PVRow get_row_count();

	PVRush::PVExtractor& get_extractor();
	PVRush::PVControllerJob_p extract();
	PVRush::PVControllerJob_p extract_from_agg_nlines(chunk_index start, chunk_index nlines);
	void wait_extract_end(PVRush::PVControllerJob_p job);

	PVRush::PVInputType_p get_input_type() const;

	inline PVAxesCombination& get_axes_combination() { return _axes_combination; }
	inline PVAxesCombination const& get_axes_combination() const { return _axes_combination; }

	void create_default_view();

	QStringList const& get_invalid_elts() const { return _inv_elts; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	void process_from_source();

	QString get_name() const { return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs); }
	QString get_format_name() const { return _extractor.get_format().get_format_name(); }
	QString get_window_name() const { return get_name() + QString(" / ") + get_format_name(); }

	PVView* current_view() const { return _current_view; }
	void select_view(PVView& view);

	PVRush::PVFormat& get_format() { return _extractor.get_format(); }
	void set_format(PVRush::PVFormat const& format);

	void set_invalid_elts_mode(bool restore_inv_elts);

	void add_column(PVAxisComputation_f f_axis, Picviz::PVAxis const& axis);

	virtual QString get_serialize_description() const { return "Source: " + get_name(); }

private:
	void add_column(Picviz::PVAxis const& axis);
	void set_views_consistent(bool cons);

protected:
	virtual void set_parent_from_ptr(PVScene* parent);
	virtual QString get_children_description() const { return "Mapped(s)"; }
	virtual QString get_children_serialize_name() const { return "mapped"; }

	void add_view(PVView_sp view);
	void set_views_id();

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
	PVRush::PVExtractor _extractor;
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	PVRush::PVInputType::list_inputs _inputs;

	PVRush::PVSourceCreator_p _src_plugin;
	PVRush::PVNraw *nraw;
	PVView* _current_view;
	bool _restore_inv_elts;
	QStringList _inv_elts;

	PVAxesCombination _axes_combination;
};

}

#endif	/* PICVIZ_PVSOURCE_H */
