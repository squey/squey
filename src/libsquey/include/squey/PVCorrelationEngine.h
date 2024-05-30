/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __SQUEY_PVCORRELATION_ENGINE_H__
#define __SQUEY_PVCORRELATION_ENGINE_H__

#include <pvbase/types.h>

#include <unordered_map>

namespace Squey
{

class PVView;

enum class PVCorrelationType {
	VALUES,
	RANGE
};

/**
 * A simple correlation descriptor
 */
struct PVCorrelation {
	const Squey::PVView* view1; //!< the origin view
	PVCol col1;                  //!< the origin column (original axis index)
	Squey::PVView* view2;       //!< the destination view
	PVCol col2;                  //!< the destination column (original axis index
	PVCorrelationType type = PVCorrelationType::VALUES;      //!< the type of correlation
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
	 * @param view the origin view1
	 * @param both_ways specify if view2 is also removed from correlations
	 */
	void remove(const Squey::PVView* view1, bool both_ways = false);

	/**
	 * Return the associated correlation for a given view
	 *
	 * @param view the origin view
	 *
	 * @ return a pointer to the correlation if the view has one
	 *          nullptr otherwise
	 */
	const PVCorrelation* correlation(const Squey::PVView* view) const;

	/**
	 * Check if the axis of a given view is the origin of an existing correlation
	 *
	 * @param view1 the origin view
	 * @param col1  the origin column
	 */
	bool exists(const Squey::PVView* view1, PVCol col1) const;

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
	Squey::PVView* process(const Squey::PVView* view);

  private:
	std::unordered_map<const Squey::PVView*, PVCorrelation> _correlations;
};

} // namespace Squey

#endif // __SQUEY_PVCORRELATION_ENGINE_H__
