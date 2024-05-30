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

#ifndef PVFIELDSPLITTERIPPARAMWIDGET_H
#define PVFIELDSPLITTERIPPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QWidget;
class QAction;
class QCheckBox;
class QRadioButton;
class QComboBox;
class QLabel;
#include <QList>

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter
{

class PVFieldSplitterIPParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterIPParamWidget();

  public:
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

  private Q_SLOTS:
	void set_ip_type(bool reset_groups_check_state = true);
	void update_child_count();

  private:
	void set_groups_check_state(bool check_all = false);

  private:
	QRadioButton* _ipv4 = nullptr;
	QRadioButton* _ipv6 = nullptr;

	QList<QCheckBox*> _cb_list;
	QList<QLabel*> _label_list;

	size_t _group_count = 0;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterIPParamWidget)
};
}

#endif // PVFIELDSPLITTERIPPARAMWIDGET_H
