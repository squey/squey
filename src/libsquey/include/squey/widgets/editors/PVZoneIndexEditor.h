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

#ifndef PVCORE_PVZONEINDEXEDITOR_H
#define PVCORE_PVZONEINDEXEDITOR_H

#include <QComboBox>
#include <QString>
#include <QWidget>

#include <pvkernel/core/PVZoneIndexType.h>

#include <squey/PVView.h>

namespace PVWidgets
{

/**
 * \class PVZoneIndexEditor
 */
class PVZoneIndexEditor : public QWidget
{
	Q_OBJECT
	Q_PROPERTY(
	    PVCore::PVZoneIndexType _zone_index READ get_zone_index WRITE set_zone_index USER true)

  public:
	explicit PVZoneIndexEditor(Squey::PVView const& view, QWidget* parent = 0);
	~PVZoneIndexEditor() override;

  public:
	PVCore::PVZoneIndexType get_zone_index() const;
	void set_zone_index(PVCore::PVZoneIndexType zone_index);

  protected:
	Squey::PVView const& _view;
	QComboBox* _first_cb;
	QComboBox* _second_cb;
};
} // namespace PVWidgets

#endif // PVCORE_PVZONEINDEXEDITOR_H
