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

#ifndef PVGUIQT_PVQNRAW_H
#define PVGUIQT_PVQNRAW_H

namespace PVRush
{
class PVNraw;
} // namespace PVRush

namespace Inendi
{
class PVSelection;
} // namespace Inendi

namespace PVGuiQt
{

class PVListUniqStringsDlg;

struct PVQNraw {
	static bool show_unique_values(Inendi::PVView& view,
	                               PVCol c,
	                               QWidget* parent = nullptr,
	                               QDialog** dialog = nullptr);
	static bool show_count_by(Inendi::PVView& view,
	                          PVCol col1,
	                          PVCol col2,
	                          Inendi::PVSelection const& sel,
	                          QWidget* parent = nullptr);
	static bool show_sum_by(Inendi::PVView& view,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
	static bool show_max_by(Inendi::PVView& view,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
	static bool show_min_by(Inendi::PVView& view,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
	static bool show_avg_by(Inendi::PVView& view,
	                        PVCol col1,
	                        PVCol col2,
	                        Inendi::PVSelection const& sel,
	                        QWidget* parent = nullptr);
};
} // namespace PVGuiQt

#endif
