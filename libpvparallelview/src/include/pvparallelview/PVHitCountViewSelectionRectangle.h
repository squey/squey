
#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWSELECTIONRECTANGLE_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWSELECTIONRECTANGLE_H

#include <pvparallelview/PVSelectionRectangle.h>

namespace Picviz
{

class PVView;

}

namespace PVParallelView
{

class PVHitCountView;

/**
 * @class PVSelectionSquareHitCountView
 *
 * a selection rectangle usable with a hit-count view.
 */
class PVHitCountViewSelectionRectangle : public PVParallelView::PVSelectionRectangle
{
public:
	/**
	 * create a selection rectangle for hit-count view
	 *
	 * @param hcv the "parent" hit-count view
	 */
	PVHitCountViewSelectionRectangle(PVHitCountView* hcv);

protected:
	/**
	 * selection commit for hit-count view
	 *
	 * @param use_selection_modifiers
	 */
	void commit(bool use_selection_modifiers) override;

	/**
	 * get the Picviz::PVView associated with the hit-count view
	 *
	 * @return the associated Picviz::PVView
	 */
	Picviz::PVView& lib_view() override;

private:
	PVHitCountView* _hcv;
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWSELECTIONRECTANGLE_H
