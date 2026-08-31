/* * MIT License
 *
 * © Squey, 2026
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

#ifndef PVGUIQT_PVANALYSISBREADCRUMB_H
#define PVGUIQT_PVANALYSISBREADCRUMB_H

#include <pvguiqt/export.h>

#include <sigc++/sigc++.h>

#include <QWidget>

class QHBoxLayout;
class QScrollArea;

namespace Squey
{
class PVRoot;
} // namespace Squey

namespace PVGuiQt
{

/**
 * \class PVAnalysisBreadcrumb
 *
 * The steps an analysis went through, laid out left to right, with the one
 * being shown picked out and the ones ahead of it dimmed.
 *
 * A crumb is a button rather than a label because the history is made of states
 * rather than of operations: landing on any of them is one call, so there is no
 * reason to make the user undo their way back to a step they can see.
 *
 * Hidden while there is nothing to come back to, so that it costs no room until
 * the first step has been taken.
 */
class PVGUIQT_EXPORT PVAnalysisBreadcrumb : public QWidget
{
	Q_OBJECT

  public:
	explicit PVAnalysisBreadcrumb(Squey::PVRoot& root, QWidget* parent = nullptr);
	~PVAnalysisBreadcrumb() override;

  public Q_SLOTS:
	void undo();
	void redo();

  Q_SIGNALS:
	/**
	 * Emitted when the steps or the position change, so that whoever holds
	 * Undo and Redo actions can enable them or not.
	 */
	void changed();

  private:
	void rebuild();
	void add_crumb(size_t index, bool is_current, bool is_ahead);

  private:
	Squey::PVRoot& _root;
	QScrollArea* _scroll = nullptr;
	QWidget* _strip = nullptr;
	QHBoxLayout* _strip_layout = nullptr;
	sigc::connection _history_changed;
};
} // namespace PVGuiQt

#endif /* PVGUIQT_PVANALYSISBREADCRUMB_H */
