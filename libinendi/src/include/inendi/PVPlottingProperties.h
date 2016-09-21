/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVPLOTTINGPROPERTIES_H
#define INENDI_PVPLOTTINGPROPERTIES_H

#include <inendi/PVPlottingFilter.h> // for PVPlottingFilter, etc

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat

#include <pvkernel/core/PVArgument.h> // for PVArgumentList

#include <pvbase/types.h> // for PVCol

#include <string> // for string

namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore
namespace PVRush
{
class PVFormat;
} // namespace PVRush

namespace Inendi
{

/**
* \class PVPlottingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class PVPlottingProperties
{
  public:
	PVPlottingProperties(PVRush::PVFormat const& fmt, PVCol idx);
	explicit PVPlottingProperties(PVRush::PVAxisFormat const& axis);
	PVPlottingProperties(std::string const& mode, PVCore::PVArgumentList args);

  public:
	// For PVPlotting
	inline void set_uptodate() { _is_uptodate = true; }
	inline void invalidate() { _is_uptodate = false; }

  public:
	PVPlottingFilter::p_type get_plotting_filter();
	void set_mode(std::string const& mode);
	void set_args(PVCore::PVArgumentList const& args);
	inline PVCore::PVArgumentList const& get_args() const { return _args; }
	inline std::string const& get_mode() const { return _mode; }
	inline bool is_uptodate() const { return _is_uptodate; }

  public:
	bool operator==(PVPlottingProperties const& org) const;

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVPlottingProperties serialize_read(PVCore::PVSerializeObject& so);

  private:
	std::string _mode;
	PVPlottingFilter::p_type _plotting_filter;
	PVCore::PVArgumentList _args;
	bool _is_uptodate = false;
};
} // namespace Inendi

#endif /* INENDI_PVPLOTTINGPROPERTIES_H */
