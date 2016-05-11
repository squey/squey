/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERFILTER_H
#define INENDI_PVLAYERFILTER_H

#include <pvbase/types.h>

#include <inendi/PVLayer_types.h>
#include <inendi/PVView_types.h>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVTag.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVPluginPresets.h>

#include <boost/function.hpp>
#include <boost/thread.hpp>

#include <pvkernel/core/PVOrderedMap.h>

namespace Inendi
{

/**
 * \class PVLayerFilter
 */
class PVLayerFilter : public PVFilter::PVFilterFunction<const PVLayer, PVLayerFilter>
{
  public:
	struct error : public std::exception {
	  public:
		using std::exception::exception;
	};

  public:
	typedef PVFilter::PVFilterFunction<const PVLayer, PVLayerFilter>::base_registrable
	    base_registrable;

  public:
	// This is used for context menu integration (in the NRAW listing)
	typedef boost::function<PVCore::PVArgumentList(PVRow, PVCol, PVCol, QString const&)>
	    ctxt_menu_f;
	// This QHash will be used for specifying a list of couple (name, function) that will be used in
	// the context menu
	typedef PVCore::PVOrderedMap<QString, ctxt_menu_f> hash_menu_function_t;

  public:
	/**
	 * Constructor
	 */
	PVLayerFilter(PVCore::PVArgumentList const& l = PVLayerFilter::default_args());

  public:
	void set_output(PVLayer* out);
	void set_view(PVView_sp const& view);
	void set_view(PVView* view) { _view = view; }
	virtual PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
	hash_menu_function_t const& get_menu_entries() const;

  public:
	virtual QString status_bar_description();
	virtual QString detailed_description();
	virtual QString menu_name() const { return registered_name(); }

  public:
	/**
	 * Show the appropriate error widget if a PVLayerFilter::error exception
	 * is thrown by the layer filter
	 */
	virtual void show_error(QWidget* /*parent*/) const { assert(false); }

  public:
	boost::thread launch_in_thread(PVLayer& layer);
	void cancel();

  protected:
	bool should_cancel();

  public:
	PVLayer const& operator()(PVLayer const& layer);

  public:
	static PVCore::PVTag<PVLayerFilter> get_tag(QString const& name);

  public:
	PVCore::PVPluginPresets<PVLayerFilter> get_presets();

  protected:
	virtual void operator()(PVLayer const&, PVLayer&)
	{
		// This method is virtual and can't be pure as we provide clone function.
		assert(false);
	};

  protected:
	void add_ctxt_menu_entry(QString menu_name, ctxt_menu_f f);

  protected:
	PVView* _view;

  private:
	PVLayer* _out_p;
	hash_menu_function_t _menu_entries;
	bool _should_cancel;

	CLASS_FILTER(PVLayerFilter)
};

typedef PVLayerFilter::p_type PVLayerFilter_p;
typedef PVLayerFilter::func_type PVLayerFilter_f;

typedef PVCore::PVClassLibrary<Inendi::PVLayerFilter>::tag PVLayerFilterTag;
typedef PVCore::PVClassLibrary<Inendi::PVLayerFilter>::list_tags PVLayerFilterListTags;
}

#endif /* INENDI_PVLAYERFILTER_H */
