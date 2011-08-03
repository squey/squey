//! \file PVLayerFilter.h
//! $Id: PVLayerFilter.h 3199 2011-06-24 07:14:57Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTER_H
#define PICVIZ_PVLAYERFILTER_H


#include <pvkernel/core/general.h>
#include <picviz/PVLayer.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <boost/function.hpp>

#include <QHash>

namespace Picviz {

/**
 * \class PVLayerFilter
 */
class LibPicvizDecl PVLayerFilter : public PVFilter::PVFilterFunction<PVLayer, PVLayerFilter> {
public:
	typedef PVFilter::PVFilterFunction<PVLayer, PVLayerFilter>::base_registrable base_registrable;
	
public:
	// This is used for context menu integration (in the NRAW listing)
	typedef boost::function<PVCore::PVArgumentList(PVRow, PVCol, QString const&)> ctxt_menu_f;
	// This QHash will be used for specifying a list of couple (name, function) that will be used in the context menu
	typedef QHash<QString, ctxt_menu_f> hash_menu_function_t;

public:
	/**
	 * Constructor
	 */
	PVLayerFilter(PVCore::PVArgumentList const& l = PVLayerFilter::default_args());//

public:
	void set_output(PVLayer* out);
	void set_view(PVView_p view);
	virtual PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
	hash_menu_function_t const& get_menu_entries() const;

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
	void add_ctxt_menu_entry(QString menu_name, ctxt_menu_f f);

protected:
	PVView_p _view;

private:
	PVLayer *_out_p;
	hash_menu_function_t _menu_entries;

	CLASS_FILTER(PVLayerFilter)
};

typedef PVLayerFilter::p_type PVLayerFilter_p;
typedef PVLayerFilter::func_type PVLayerFilter_f;

// For this wto work under windows, wez need to export here the PVFilterLibrary for PVLayerFilter
#ifdef WIN32
LibPicvizDeclExplicitTempl PVCore::PVClassLibrary<Picviz::PVLayerFilter>;
#endif

}

#endif /* PICVIZ_PVLAYERFILTER_H */

