/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSELECTIONFILTER_H
#define INENDI_PVSELECTIONFILTER_H


#include <pvkernel/core/general.h>
#include <inendi/PVSelection.h>
#include <inendi/PVView.h>

#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVArgument.h>

namespace Inendi {

/**
 * \class PVSelectionFilter
 */
class PVSelectionFilter : public PVFilter::PVFilterFunction<PVSelection, PVSelectionFilter> {
public:
	typedef PVFilter::PVFilterFunction<PVSelection, PVSelectionFilter>::base_registrable base_registrable;
public:
	/**
	 * Constructor
	 */
	PVSelectionFilter(PVCore::PVArgumentList const& l = PVSelectionFilter::default_args());

public:
	void set_output(PVSelection* out);
	void set_view(PVView const& view);

public:
	PVSelection& operator()(PVSelection& layer);
	PVSelection& operator_sameout(PVSelection &in);
	PVSelection& operator_differentout(PVSelection &in);

public:
	virtual void operator()(PVSelection &in, PVSelection &out);

protected:
	PVView const* _view;

private:
	PVSelection *_out_p;

	CLASS_FILTER(Inendi::PVSelectionFilter)
};

typedef std::shared_ptr<PVSelectionFilter> PVSelectionFilter_p;

}

#endif /* INENDI_PVSELECTIONFILTER_H */

