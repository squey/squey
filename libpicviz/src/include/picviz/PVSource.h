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

namespace Picviz {

/**
 * \class PVSource
 */
class LibPicvizDecl PVSource {
	friend class PVCore::PVSerializeObject;
public:
	typedef PVSource_p p_type;
public:
	PVSource(PVScene_p parent);
	~PVSource();

	PVScene_p tparent;
	PVRoot_p root;
	/* PVRush::PVFormat *format; */

	PVRush::File *file;

	PVRow limit_min;
	PVRow limit_max;

	QString logtype;

	PVAxesCombination axes_combination;

	PVRush::PVNraw *nraw;

	/* Functions */
	PVCol get_column_count();

	PVRush::PVNraw::nraw_table& get_qtnraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;

	PVRush::PVNraw& get_rushnraw();
	const PVRush::PVNraw& get_rushnraw() const;

	const PVRush::PVNraw::nraw_trans_table& get_trans_nraw() const;
	void clear_trans_nraw();
	PVRow get_row_count();
	QString get_value(PVRow row, PVCol col);
	void set_limits(PVRow min, PVRow max);


	PVRush::PVControllerJob_p files_append(PVRush::PVFormat const& format, PVRush::PVSourceCreator_p sc, PVRush::PVInputType::list_inputs inputs);
	void files_append_noextract(PVRush::PVFormat const& format, PVRush::PVSourceCreator_p sc, PVRush::PVInputType::list_inputs inputs);
	PVRush::PVExtractor& get_extractor();

	PVRush::PVControllerJob_p reextract();

	PVRush::PVInputType_p get_input_type();

protected:
	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);
	void serialize_write(PVCore::PVSerializeObject& so);

	PVSERIALIZEOBJECT_SPLIT

protected:
	void set_format(PVRush::PVFormat const& format);
	PVRush::PVRawSourceBase_p create_extractor_source(QString type, QString filename, PVRush::PVFormat const& format);
	PVRush::PVExtractor _extractor;

private:
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	PVRush::PVInputType::list_inputs _inputs;
	PVRush::PVSourceCreator_p _src_plugin;
};

}

#endif	/* PICVIZ_PVSOURCE_H */
