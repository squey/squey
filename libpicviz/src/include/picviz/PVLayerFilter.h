//! \file PVLayerFilter.h
//! $Id: PVLayerFilter.h 3199 2011-06-24 07:14:57Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTER_H
#define PICVIZ_PVLAYERFILTER_H


#include <pvcore/general.h>
#include <picviz/PVLayer.h>
#include <picviz/PVView.h>

#include <pvfilter/PVFilterFunction.h>
#include <pvfilter/PVFilterLibrary.h>
#include <pvfilter/PVArgument.h>

namespace Picviz {

/**
 * \class PVLayerFilter
 */
class LibPicvizDecl PVLayerFilter : public PVFilter::PVFilterFunction<PVLayer, PVLayerFilter> {
public:
	typedef PVFilter::PVFilterFunction<PVLayer, PVLayerFilter>::base_registrable base_registrable;
public:
	/**
	 * Constructor
	 */
	PVLayerFilter(PVFilter::PVArgumentList const& l = PVLayerFilter::default_args());//

public:
	void set_output(PVLayer* out);
	void set_view(PVView_p view);
	virtual PVFilter::PVArgumentList get_default_args_for_view(PVView const& view);

public:
	virtual QString status_bar_description();
	virtual QString detailed_description();

public:
	PVLayer& operator()(PVLayer& layer);
	PVLayer& operator_sameout(PVLayer &in);
	PVLayer& operator_differentout(PVLayer &in);

protected:
	virtual void operator()(PVLayer &in, PVLayer &out);

protected:
	PVView_p _view;

private:
	PVLayer *_out_p;

	CLASS_FILTER(PVLayerFilter)
};

typedef PVLayerFilter::p_type PVLayerFilter_p;
typedef PVLayerFilter::func_type PVLayerFilter_f;

// For this wto work under windows, wez need to export here the PVFilterLibrary for PVLayerFilter
#ifdef WIN32
picviz_FilterLibraryDecl PVFilter::PVFilterLibrary<Picviz::PVLayerFilter>;
#endif

}

#endif /* PICVIZ_PVLAYERFILTER_H */

