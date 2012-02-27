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

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/general.h>
#include <picviz/PVMappingProperties.h>
#include <picviz/PVPtrObjects.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVMandatoryMappingFilter.h>

#include <boost/shared_ptr.hpp>

namespace Picviz {

class PVMapped;

/**
 * \class PVMapping
 */
class LibPicvizDecl PVMapping
{
	friend class PVMapped;
	friend class PVCore::PVSerializeObject;
public:
	typedef boost::shared_ptr<PVMapping> p_type;

public:
	PVMapping(PVSource* parent);
	~PVMapping();

protected:
	// For serialization
	PVMapping() { };
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

	// For PVMapped
	void set_source(PVSource* src);
	void set_uptodate_for_col(PVCol j);
	void invalidate_all();
	void validate_all();
	void add_column(PVMappingProperties const& props);

public:
	float get_position(int column, QString const& value);
	bool is_uptodate() const;

public:
	// Parents
	PVSource* get_source_parent();
	PVRoot* get_root_parent();
	const PVSource* get_source_parent() const;
	const PVRoot* get_root_parent() const;
	PVRush::PVFormat_p get_format() const;

public:
	// NRAW
	PVRush::PVNraw::nraw_table& get_qtnraw();
	PVRush::PVNraw::nraw_trans_table const& get_trans_nraw() const;
	void clear_trans_nraw();
	const PVRush::PVNraw::nraw_table& get_qtnraw() const;

public:
	// Column properties
	PVMappingFilter::p_type get_filter_for_col(PVCol col);
	QString const& get_type_for_col(PVCol col) const;
	QString const& get_mode_for_col(PVCol col) const;
	QString get_group_key_for_col(PVCol col) const;
	PVMappingProperties const& get_properties_for_col(PVCol col) const { assert(col < columns.size()); return columns.at(col); }
	PVMappingProperties& get_properties_for_col(PVCol col) { assert(col < columns.size()); return columns[col]; }
	bool is_col_uptodate(PVCol j) const;
	PVCol get_number_cols() const { return columns.size(); }

	QString const& get_name() const { return _name; }
	void set_name(QString const& name) { _name = name; }

	void reset_from_format(PVRush::PVFormat const& format);
	void set_default_args(PVRush::PVFormat const& format);

public:
	// Mandatory parameters
	mandatory_param_map const& get_mandatory_params_for_col(PVCol col) const;
	mandatory_param_map& get_mandatory_params_for_col(PVCol col);

protected:
	QVector<mandatory_param_map> _mandatory_filters_values;
	QList<PVMappingProperties> columns;

	PVSource* source;
	PVRoot* root;

	QString _name;
};

typedef PVMapping::p_type PVMapping_p;

}

#endif	/* PICVIZ_PVMAPPING_H */
