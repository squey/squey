//! \file PVSource.h
//! $Id: PVSource.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSOURCE_H
#define PICVIZ_PVSOURCE_H

#include <QString>
#include <QList>
#include <QStringList>
#include <QVector>

#include <pvkernel/core/general.h>
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
class LibPicvizDecl PVSource: public boost::enable_shared_from_this<PVSource>
{
	friend class PVCore::PVSerializeObject;
	friend class PVScene;
	friend class PVView;
	friend class PVPlotted;
public:
	typedef PVSource_p p_type;
	typedef QList<PVView_p> list_views_t;
	typedef QList<PVMapped_p> list_mapped_t;
public:
	PVSource(PVRush::PVInputType::list_inputs const& inputs, PVRush::PVSourceCreator_p sc, PVRush::PVFormat format);
	~PVSource();
protected:
	PVSource();
	PVSource(const PVSource& org);

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

	void add_mapped(PVMapped_p mapped);
	void create_default_view();

	// Parents
	inline PVRoot* get_root() const { return root; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	void process_from_source(bool keep_view_infos);

	QString get_name() const { return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs); }
	QString get_format_name() const { return _extractor.get_format().get_format_name(); }
	QString get_window_name() const { return get_name() + QString(" / ") + get_format_name(); }

	list_views_t const& get_views() const { return _views; }
	list_mapped_t const& get_mappeds() const { return _mappeds; }

	PVView_p current_view() const { return _current_view; }
	void select_view(PVView_p view) { assert(_views.contains(view)); _current_view = view; }

	PVRush::PVFormat& get_format() { return _extractor.get_format(); }
	void set_format(PVRush::PVFormat const& format);

	void add_column(PVAxisComputation_f f_axis, Picviz::PVAxis const& axis);

private:
	void add_column(Picviz::PVAxis const& axis);
	void set_views_consistent(bool cons);

protected:
	// For PVScene
	void set_parent(PVScene* parent);
	// For PVPlotted
	void add_view(PVView_p view);

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
	PVScene* tparent;
	PVRoot* root;

	PVRush::PVExtractor _extractor;
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	PVRush::PVInputType::list_inputs _inputs;
	list_views_t _views;
	list_mapped_t _mappeds;
	PVRush::PVSourceCreator_p _src_plugin;
	PVRush::PVNraw *nraw;
	PVView_p _current_view;

	PVAxesCombination _axes_combination;
};

}

#endif	/* PICVIZ_PVSOURCE_H */
