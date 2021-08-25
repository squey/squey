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

#ifndef PVLAYERSTACKWIDGET_H
#define PVLAYERSTACKWIDGET_H

#include <QDialog>
#include <QToolBar>
#include <QWidget>

namespace Inendi
{
class PVView;
} // namespace Inendi

namespace PVGuiQt
{

class PVLayerStackModel;
class PVLayerStackView;

/**
 *  \class PVLayerStackWidget
 */
class PVLayerStackWidget : public QWidget
{
	Q_OBJECT

  public:
	explicit PVLayerStackWidget(Inendi::PVView& lib_view, QWidget* parent = nullptr);

  public:
	PVLayerStackView* get_layer_stack_view() const { return _layer_stack_view; }

  private:
	void create_actions(QToolBar* toolbar);
	PVLayerStackModel* ls_model();

  private Q_SLOTS:
	void delete_layer();
	void duplicate_layer();
	void move_down();
	void move_up();
	void new_layer();
	void export_all_layers();

  private:
	PVLayerStackView* _layer_stack_view;
};
} // namespace PVGuiQt

#endif // PVLAYERSTACKWIDGET_H
