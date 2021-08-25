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

#ifndef INENDI_PVLAYERFILTERMULTIPLESEARCH_H
#define INENDI_PVLAYERFILTERMULTIPLESEARCH_H

#include <inendi/PVLayer.h>
#include <inendi/PVLayerFilter.h>

#include <pvcop/db/exceptions/partially_converted_error.h>

namespace Inendi
{

/**
 * \class PVLayerFilterMultipleSearch
 */
class PVLayerFilterMultipleSearch : public PVLayerFilter
{
  public:
	PVLayerFilterMultipleSearch(
	    PVCore::PVArgumentList const& l = PVLayerFilterMultipleSearch::default_args());

  public:
	void operator()(PVLayer const& in, PVLayer& out) override;
	PVCore::PVArgumentKeyList get_args_keys_for_preset() const override;
	QString menu_name() const override { return "Text Search/Multiple values"; }

  public:
	static PVCore::PVArgumentList
	search_value_menu(PVRow row, PVCombCol col, PVCol org_col, QString& v);
	static PVCore::PVArgumentList
	search_using_value_menu(PVRow row, PVCombCol col, PVCol org_col, QString& v);
	static PVCore::PVArgumentList search_menu(PVRow row, PVCombCol col, PVCol org_col, QString& v);

	void show_error(QWidget* parent) const override;

  private:
	using strings_t = std::remove_reference<decltype(
	    ((pvcop::db::exception::partially_converted_error*)nullptr)->bad_values())>::type;
	strings_t _unconverted_values;

	CLASS_FILTER(Inendi::PVLayerFilterMultipleSearch)
};
}

#endif
