/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVAXIS_H
#define INENDI_PVAXIS_H

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat, etc

#include <pvkernel/core/PVArgument.h> // for PVArgumentList

#include <stdexcept> // for runtime_error

namespace PVCore
{
class PVSerializeObject;
}

namespace Inendi
{

/**
 * Exception raised when mapping/plotting combination is invalid.
 */
struct InvalidPlottingMapping : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

/**
 * \class PVAxis
 */
class PVAxis : public PVRush::PVAxisFormat
{
	friend class PVCore::PVSerializeObject;

  public:
	/**
	 * Constructor
	 */
	PVAxis()
	    : PVRush::PVAxisFormat(-1){}; // We have to keep this Ugly constructor as we use QVector
	                                  // which perform a lot of
	                                  // default construction
	explicit PVAxis(PVRush::PVAxisFormat axis_format);

  public:
	PVCore::PVArgumentList const& get_args_mapping() const { return _args_mapping; }
	PVCore::PVArgumentList const& get_args_plotting() const { return _args_plotting; }

  private:
	static PVCore::PVArgumentList args_from_node(node_args_t const& args_str,
	                                             PVCore::PVArgumentList const& def_args);

  private:
	PVCore::PVArgumentList _args_mapping;
	PVCore::PVArgumentList _args_plotting;
};
}

#endif /* INENDI_PVAXIS_H */
