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

#ifndef PVCORE_PVLAYERENUMEDITOR_H
#define PVCORE_PVLAYERENUMEDITOR_H

#include <QComboBox>

#include <inendi/PVView.h>

namespace PVWidgets
{

class PVLayerEnumEditor : public QComboBox
{
	Q_OBJECT
	Q_PROPERTY(Inendi::PVLayer* _layer READ get_layer WRITE set_layer USER true)

  public:
	explicit PVLayerEnumEditor(Inendi::PVView const& view, QWidget* parent = 0);
	~PVLayerEnumEditor() override;

  public:
	Inendi::PVLayer* get_layer() const;
	void set_layer(Inendi::PVLayer* l);

  protected:
	Inendi::PVView const& _view;
};
} // namespace PVWidgets

#endif
