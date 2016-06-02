/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef __INENDI_PVCORRELATION_ENGINE_H__
#define __INENDI_PVCORRELATION_ENGINE_H__

#include <pvbase/types.h>

#include <tuple>
#include <unordered_map>

#include <pvlogger.h>

namespace Inendi
{

class PVView;

/**
 * A simple correlation descriptor
 */
struct PVCorrelation {
	const Inendi::PVView* view1; //!< the origin view
	PVCol col1;                  //!< the origin column
	Inendi::PVView* view2;       //!< the destination view
	PVCol col2;                  //!< the destination column
};

/**
 * Current limitations :
 *     (1) Supported types are limited to "integer" and "ipv4"
 *     (2) Only one column binding per correlation is supported
 *     (3) Only direct correlations are supported
 *     (4) No manual selection propagation support
 *     (5) No serialization support
 */
class PVCorrelationEngine
{
  public:
	/**
	 * Activate a new correlation
	 *
	 * @param correlation a given correlation
	 */
	void add(const PVCorrelation& correlation);

	/**
	 * Deactivate an existing correlation
	 *
	 * @param view1 the origin view
	 */
	void remove(const Inendi::PVView* view1);

	/**
	 * Return the associated correlation for a given view
	 *
	 * @param view the origin view
	 *
	 * @ return a pointer to the correlation if the view has one
	 *          nullptr otherwise
	 */
	const PVCorrelation* correlation(const Inendi::PVView* view) const;

	/**
	 * Check if the axis of a given view is the origin of an existing correlation
	 *
	 * @param view1 the origin view
	 * @param col1  the origin column
	 */
	bool exists(const Inendi::PVView* view1, PVCol col1) const;

	/**
	 * Check if a correlation exists between a pair of view and columns
	 *
	 * @param correlation a given correlation
	 */
	bool exists(const PVCorrelation& correlation) const;

	/**
	 * Propagate the selection of a view through its correlation
	 *
	 * @param view the origin view
	 */
	Inendi::PVView* process(const Inendi::PVView* view);

  private:
	std::unordered_map<const Inendi::PVView*, PVCorrelation> _correlations;
};

} // namespace Inendi

#endif // __INENDI_PVCORRELATION_ENGINE_H__
