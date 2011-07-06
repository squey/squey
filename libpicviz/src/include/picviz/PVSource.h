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

#include <pvcore/general.h>

#include <pvrush/PVFormat.h>
#include <pvrush/PVFile.h>
#include <pvrush/PVNraw.h>

#include <picviz/PVAxesCombination.h>

#include <pvrush/PVExtractor.h> 
#include <pvrush/PVFormat.h>
#include <pvrush/PVInputType.h>
#include <pvrush/PVSourceCreator.h>

#include <picviz/PVScene.h>
#include <picviz/PVRoot.h>

#include <boost/shared_ptr.hpp>

namespace Picviz {

/**
 * \class PVSource
 */
class LibExport PVSource {
public:
	typedef boost::shared_ptr<PVSource> p_type;
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
	const PVRush::PVNraw::nraw_trans_table& get_trans_nraw() const;
	void clear_trans_nraw();
	PVRow get_row_count();
	QString get_value(PVRow row, PVCol col);
	void set_limits(PVRow min, PVRow max);

	PVRush::PVControllerJob_p files_append(PVRush::PVFormat const& format, PVRush::PVSourceCreator_p sc, PVRush::PVInputType::list_inputs inputs);
	PVRush::PVExtractor& get_extractor();

protected:
	PVFilter::PVRawSourceBase_p create_extractor_source(QString type, QString filename, PVRush::PVFormat const& format);
	PVRush::PVExtractor _extractor;

private:
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
};

typedef PVSource::p_type PVSource_p;

}

#endif	/* PICVIZ_PVSOURCE_H */
