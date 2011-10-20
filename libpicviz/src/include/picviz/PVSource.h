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
public:
	typedef PVSource_p p_type;
	typedef QList<PVView_p> list_views_t;
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

	QString const& get_value(PVRow row, PVCol col) const;
	PVRow get_row_count();

	PVRush::PVExtractor& get_extractor();
	PVRush::PVControllerJob_p extract();

	PVRush::PVInputType_p get_input_type() const;

	inline PVAxesCombination& get_axes_combination() { return _axes_combination; }
	inline PVAxesCombination const& get_axes_combination() const { return _axes_combination; }

	void add_view(PVView_p view);

	// Parents
	inline PVRoot_p get_root() const { return root; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	void process_from_source(bool keep_layers);

	QString get_name() { return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs); }
	QString get_format_name() { return _extractor.get_format().get_format_name(); }

	list_views_t const& get_views() const { return _views; }

protected:
	// For PVScene
	void set_parent(PVScene_p parent);
	// For PVView objects when they are being deleted
	bool del_view(const PVView* view);

protected:
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);

	PVSERIALIZEOBJECT_SPLIT

private:
	void set_format(PVRush::PVFormat const& format);
	PVRush::PVRawSourceBase_p create_extractor_source(QString type, QString filename, PVRush::PVFormat const& format);
	void files_append_noextract();
	void init();

private:
	PVScene_p tparent;
	PVRoot_p root;

	PVRush::PVExtractor _extractor;
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	PVRush::PVInputType::list_inputs _inputs;
	list_views_t _views;
	PVRush::PVSourceCreator_p _src_plugin;
	PVRush::PVNraw *nraw;

	PVAxesCombination _axes_combination;
};

}

#endif	/* PICVIZ_PVSOURCE_H */
