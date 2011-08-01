//! \file PVSelectionFilter.h
//! $Id: PVSelectionFilter.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSELECTIONFILTER_H
#define PICVIZ_PVSELECTIONFILTER_H


#include <pvcore/general.h>
#include <picviz/PVSelection.h>
#include <picviz/PVView.h>

#include <pvfilter/PVFilterFunction.h>
#include <pvfilter/PVFilterLibrary.h>
#include <pvcore/PVArgument.h>

namespace Picviz {

/**
 * \class PVSelectionFilter
 */
class LibPicvizDecl PVSelectionFilter : public PVFilter::PVFilterFunction<PVSelection, PVSelectionFilter> {
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

	CLASS_FILTER(Picviz::PVSelectionFilter)
};

typedef boost::shared_ptr<PVSelectionFilter> PVSelectionFilter_p;

// For this wto work under windows, wez need to export here the PVFilterLibrary for PVLayerFilter
#ifdef WIN32
LibPicvizDeclExplicitTempl PVFilter::PVFilterLibrary<Picviz::PVSelectionFilter>;
#endif

}

#endif /* PICVIZ_PVSELECTIONFILTER_H */

