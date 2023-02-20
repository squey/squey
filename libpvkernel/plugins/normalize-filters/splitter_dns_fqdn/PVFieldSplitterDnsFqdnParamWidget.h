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

#ifndef PVFIELDSPLITTERDNSFQDNPARAMWIDGET_H
#define PVFIELDSPLITTERDNSFQDNPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

#include <QAction>
#include <QWidget>
#include <QCheckBox>

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter
{

class PVFieldSplitterDnsFqdnParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterDnsFqdnParamWidget();

	size_t force_number_children() override { return 0; }

  public:
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

  private:
	void update_args(PVCore::PVArgumentList& args);

  private Q_SLOTS:
	void split_cb_changed(int state);
	void rev_cb_changed(int state);

  private:
	QWidget* _param_widget = nullptr;
	QCheckBox* _split_cb[6] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
	QCheckBox* _rev_cb[3] = { nullptr, nullptr, nullptr };

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterDnsFqdnParamWidget)
};
}

#endif // PVFIELDSPLITTERDNSFQDNPARAMWIDGET_H
