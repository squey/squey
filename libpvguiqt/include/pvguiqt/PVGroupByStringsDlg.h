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

#ifndef __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
#define __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvguiqt/PVListUniqStringsDlg.h>

#include <QAbstractListModel>
#include <QMenu>

namespace PVGuiQt
{

class PVGroupByStringsDlg : public PVAbstractListStatsDlg
{
  public:
	PVGroupByStringsDlg(Squey::PVView& view,
	                    PVCol c1,
	                    PVCol c2,
	                    const create_model_f& f,
	                    const Squey::PVSelection& sel,
	                    bool counts_are_integers,
	                    QWidget* parent = nullptr)
	    : PVAbstractListStatsDlg(view, c1, f, counts_are_integers, parent)
	    , _col2(c2)
	    , _col2_name(view.get_parent<Squey::PVSource>().get_format().get_axes().at(c2).get_name())
	    , _sel(sel)
	{
		_ctxt_menu->addSeparator();
		_act_details = new QAction("Show details", _ctxt_menu);
		_ctxt_menu->addAction(_act_details);
	}

	bool process_context_menu(QAction* act) override;

	PVStatsModel*
	details_create_model(const Squey::PVView& view, PVCol c, Squey::PVSelection const& sel);

  private:
	PVCol _col2;
	QString _col2_name;
	Squey::PVSelection _sel; //!< Store selection to be able to compute 'details'
	QAction* _act_details;    //!< Action to show details
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVCOUNTBYSTRINGSDLG_H__
