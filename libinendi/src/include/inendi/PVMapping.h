/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVMAPPING_H
#define INENDI_PVMAPPING_H

#include <QList>
#include <QLibrary>
#include <QVector>

#include <pvkernel/core/PVDataTreeObject.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVMappingProperties.h>
#include <inendi/PVMappingFilter.h>

#include <memory>

namespace Inendi
{

class PVMapped;
class PVSource;

/**
 * \class PVMapping
 */
class PVMapping
{
	friend class PVMapped;
	friend class PVCore::PVSerializeObject;

  public:
	typedef std::shared_ptr<PVMapping> p_type;

  public:
	PVMapping(PVMapped* mapped);

  protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

	// For PVMapped
	void set_parent(PVSource* src);
	void set_uptodate_for_col(PVCol j);

	void add_column(PVMappingProperties const& props);

  public:
	float get_position(int column, QString const& value);
	bool is_uptodate() const;

  public:
	void set_mapped(PVMapped* mapped) { _mapped = mapped; }
	PVMapped* get_mapped() { return _mapped; }

	void set_source(PVSource* src);

	PVRush::PVFormat const& get_format() const;

  public:
	// Column properties
	PVMappingFilter::p_type get_filter_for_col(PVCol col);
	QString const& get_mode_for_col(PVCol col) const;
	PVMappingProperties const& get_properties_for_col(PVCol col) const
	{
		assert(col < columns.size());
		return columns.at(col);
	}
	PVMappingProperties& get_properties_for_col(PVCol col)
	{
		assert(col < columns.size());
		return columns[col];
	}
	bool is_col_uptodate(PVCol j) const;
	PVCol get_number_cols() const { return columns.size(); }

	std::string const& get_name() const { return _name; }
	void set_name(std::string const& name) { _name = name; }

	void set_default_args(PVRush::PVFormat const& format);

  protected:
	QList<PVMappingProperties> columns;

	std::string _name;
	PVMapped* _mapped;
};

typedef PVMapping::p_type PVMapping_p;
}

#endif /* INENDI_PVMAPPING_H */
