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

#ifndef __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
#define __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__

#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvkernel/core/PVProgressBox.h>

#include <pvguiqt/PVStatsModel.h>

#include <pvcop/db/array.h>
#include <pvcop/db/algo.h>

namespace PVGuiQt
{

class PVListUniqStringsDlg : public PVAbstractListStatsDlg
{
  public:
	PVListUniqStringsDlg(Inendi::PVView& view,
	                     PVCol c,
	                     const create_model_f& f,
	                     QWidget* parent = nullptr)
	    : PVAbstractListStatsDlg(view,
	                             c,
	                             f,
	                             true, /* counts_are_integer */
	                             parent)
	{
		QString col1_name =
		    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(c).get_name();
		setWindowTitle("Distinct values of axe '" + col1_name + "'");
	}
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVLISTUNIQSTRINGSDLG_H__
