/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVPLOTTING_H
#define INENDI_PVPLOTTING_H

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <inendi/PVPlottingProperties.h>
#include <inendi/PVPlottingFilter.h>

#include <memory>
#include <list>

namespace Inendi
{

class PVMapped;
class PVPlotted;

/**
 * \class PVPlotting
 */
class PVPlotting
{
	friend class PVCore::PVSerializeObject;
	friend class PVPlotted;

  public:
	typedef std::shared_ptr<PVPlotting> p_type;

  public:
	/**
	 * Constructor
	 */
	PVPlotting(PVPlotted* mapped);

	/**
	 * Destructor
	 */
	~PVPlotting();

  protected:
	// Serialization
	PVPlotting();
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

	// For PVPlotted
	void set_uptodate_for_col(PVCol j);
	void invalidate_column(PVCol j);

  public:
	// Parents

	/**
	 * Gets the associated format
	 */
	PVRush::PVFormat const& get_format() const;

	PVPlotted* get_plotted() { return _plotted; }
	PVPlotted const* get_plotted() const { return _plotted; }

	bool is_uptodate() const;

  public:
	// Data access
	Inendi::PVPlottingFilter::p_type get_filter_for_col(PVCol col);
	PVPlottingProperties const& get_properties_for_col(PVCol col) const
	{
		assert((size_t)col < _columns.size());
		auto begin = _columns.begin();
		std::advance(begin, col);
		return *begin;
	}
	PVPlottingProperties& get_properties_for_col(PVCol col)
	{
		assert((size_t)col < _columns.size());
		auto begin = _columns.begin();
		std::advance(begin, col);
		return *begin;
	}
	bool is_col_uptodate(PVCol j) const;

	std::string const& get_name() const { return _name; }
	void set_name(std::string const& name) { _name = name; }

  protected:
	std::list<PVPlottingProperties> _columns;

	PVPlotted* _plotted;
	std::string _name;
};

typedef PVPlotting::p_type PVPlotting_p;
}

#endif /* INENDI_PVPLOTTING_H */
