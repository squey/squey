//! \file PVMapping.h
//! $Id: PVMapping.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVMAPPING_H
#define PICVIZ_PVMAPPING_H

#include <QList>
#include <QLibrary>
#include <QVector>

#include <pvrush/PVFormat.h>
#include <pvrush/PVNraw.h>

#include <picviz/general.h>
#include <picviz/PVMappingProperties.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVMandatoryMappingFilter.h>

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>


namespace Picviz {

#ifndef picviz_mapping_function
#define picviz_mapping_function "picviz_mapping_function"
#endif

/**
 * \class PVMapping
 */
class LibExport PVMapping : public boost::enable_shared_from_this<PVMapping> {
public:
	typedef boost::shared_ptr<PVMapping> p_type;

public:
	PVMapping(PVSource_p parent);
	~PVMapping();

	PVSource_p source;
	PVRoot_p root;
	QList<PVMappingProperties> columns;

	PVRush::PVFormat *get_format() const;
	float get_position(int column, QString const& value);
	PVRush::PVNraw::nraw_table& get_qtnraw();
	PVRush::PVNraw::nraw_trans_table const& get_trans_nraw() const;
	void clear_trans_nraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;
	PVSource_p get_source_parent();

	PVMappingFilter::p_type get_filter_for_col(PVCol col);

	mandatory_param_map const& get_mandatory_params_for_col(PVCol col) const;
	mandatory_param_map& get_mandatory_params_for_col(PVCol col);

protected:
	QVector<mandatory_param_map> _mandatory_filters_values;
};

typedef PVMapping::p_type PVMapping_p;

}

#endif	/* PICVIZ_PVMAPPING_H */
