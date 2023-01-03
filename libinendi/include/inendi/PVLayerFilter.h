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

#ifndef INENDI_PVLAYERFILTER_H
#define INENDI_PVLAYERFILTER_H

#include <pvkernel/filter/PVFilterFunction.h> // for CLASS_FILTER

#include <pvkernel/core/PVArgument.h>     // for PVArgumentList
#include <pvkernel/core/PVClassLibrary.h> // for PVClassLibrary, etc
#include <pvkernel/core/PVFunctionArgs.h>
#include <pvkernel/core/PVOrderedMap.h>    // for PVOrderedMap
#include <pvkernel/core/PVPluginPresets.h> // for PVPluginPresets
#include <pvkernel/core/PVRegistrableClass.h>

#include <pvbase/types.h> // for PVCol, PVRow

#include <QString> // for QString

#include <cassert>   // for assert
#include <exception> // for exception

class QWidget;

namespace Inendi
{

class PVLayer;
class PVView;

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
	typedef std::function<PVCore::PVArgumentList(PVRow, PVCombCol, PVCol, QString&)> ctxt_menu_f;
	// This QHash will be used for specifying a list of couple (name, function) that will be used in
	// the context menu
	typedef PVCore::PVOrderedMap<QString, ctxt_menu_f> hash_menu_function_t;

  public:
	/**
	 * Constructor
	 */
	explicit PVLayerFilter(PVCore::PVArgumentList const& l = PVLayerFilter::default_args());

  public:
	void set_output(PVLayer* out);
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
	void cancel();

  protected:
	bool should_cancel();

  public:
	PVLayer const& operator()(PVLayer const& layer) override;

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
} // namespace Inendi

#endif /* INENDI_PVLAYERFILTER_H */
