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

#ifndef PVINPUTTYPEMENUENTRIES_H
#define PVINPUTTYPEMENUENTRIES_H

#include <QMenu>
#include <QObject>
#include <QBoxLayout>

#include <pvkernel/rush/PVInputType.h>

namespace PVGuiQt
{

class PVInputTypeMenuEntries
{
  public:
	static void add_inputs_to_menu(QMenu* menu, QObject* parent, const char* slot);
	static void add_inputs_to_layout(QBoxLayout* layout, QObject* parent, const char* slot);
	static PVRush::PVInputType_p input_type_from_action(QAction* action);
};
} // namespace PVGuiQt

#endif
