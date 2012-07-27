/**
 * \file PVLayerFilter.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERFILTER_H
#define PICVIZ_PVLAYERFILTER_H


#include <pvkernel/core/general.h>
#include <picviz/PVLayer_types.h>
#include <picviz/PVView_types.h>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVTag.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVPluginPresets.h>

#include <boost/function.hpp>
#include <boost/thread.hpp>

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
	PVLayerFilter(PVCore::PVArgumentList const& l = PVLayerFilter::default_args());

public:
	void set_output(PVLayer* out);
	void set_view(PVView_p view);
	virtual PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
	hash_menu_function_t const& get_menu_entries() const;

public:
	virtual QString status_bar_description();
	virtual QString detailed_description();
	virtual QString menu_name() const { return registered_name(); }

public:
	// Helper function for tags

public:
	boost::thread launch_in_thread(PVLayer& layer);
	void cancel();

protected:
	bool should_cancel();

public:
	PVLayer& operator()(PVLayer& layer);
	PVLayer& operator_sameout(PVLayer &in);
	PVLayer& operator_differentout(PVLayer &in);

public:
	static PVCore::PVTag<PVLayerFilter> get_tag(QString const& name);

public:
	PVCore::PVPluginPresets<PVLayerFilter> get_presets();

protected:
	virtual void operator()(PVLayer &in, PVLayer &out);

protected:
	void add_ctxt_menu_entry(QString menu_name, ctxt_menu_f f);

protected:
	PVView_p _view;

private:
	PVLayer *_out_p;
	hash_menu_function_t _menu_entries;
	bool _should_cancel;

	CLASS_FILTER(PVLayerFilter)
};

typedef PVLayerFilter::p_type PVLayerFilter_p;
typedef PVLayerFilter::func_type PVLayerFilter_f;

// For this to work under windows, we need to export here the PVFilterLibrary for PVLayerFilter
#ifdef WIN32
LibPicvizDeclExplicitTempl PVCore::PVClassLibrary<Picviz::PVLayerFilter>;
LibPicvizDeclExplicitTempl PVCore::PVTag<Picviz::PVLayerFilter>;
#endif

typedef PVCore::PVClassLibrary<Picviz::PVLayerFilter>::tag PVLayerFilterTag;
typedef PVCore::PVClassLibrary<Picviz::PVLayerFilter>::list_tags PVLayerFilterListTags;

}

#endif /* PICVIZ_PVLAYERFILTER_H */
