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

#ifndef PVFILTER_PVFIELDSPLITTERLENGTHPARAMWIDGET_H
#define PVFILTER_PVFIELDSPLITTERLENGTHPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QPushButton;

namespace PVFilter
{

class PVFieldSplitterLengthParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterLengthParamWidget();

  public:
	QWidget* get_param_widget() override;
	QAction* get_action_menu(QWidget* parent) override;

  public Q_SLOTS:
	void update_length(int value);
	void update_side(bool state);

  protected:
	void set_button_text(bool state);

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterLengthParamWidget)

  private:
	QPushButton* _side = nullptr;
};
}

#endif // PVFILTER_PVFIELDSPLITTERLENGTHPARAMWIDGET_H
