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

#ifndef __PVWIDGETS_PVQUERYBUILDER_H__
#define __PVWIDGETS_PVQUERYBUILDER_H__

#include <qstring.h>
#include <qtmetamacros.h>
#include <QApplication>
#include <string>
#include <utility>
#include <vector>
#ifdef QT_WEBKIT
class QWebView;
#else
class QWebEngineView;
#endif

#include <QWidget>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace PVWidgets
{

/** This class provides a wrapper around jQuery plugin QueryBuilder
 * (http://mistic100.github.io/jQuery-QueryBuilder/) using Qt chromium
 * based web engine (QWebEngine).
 *
 * The concept "filters" represents the choices given to the user (name, type and operators)
 * The concept "rules" represents the query of the user (for serialization and convertion)
 */
class PVQueryBuilder : public QWidget
{
	Q_OBJECT;

  private:
	using columns_t = std::vector<std::pair<std::string, std::string>>;

  public:
	explicit PVQueryBuilder(QWidget* parent = nullptr);

  public:
	/** Set the widget filters
	 *
	 * Note that as the jQuery widget doesn't expose any method
	 * to change the filters at runtime, this call causes the
	 * whole jQuery widget to be recreated using the reinit method.
	 *
	 * @param filters a valid JSON string containing the filters
	 */
	void set_filters(const std::string& filters);

	/** Convenient method to set the widget filters using a list
	 * of columns with their type.
	 *
	 * @param cols the list of columns with their type
	 */
	void set_filters(const columns_t& cols);

  public:
	/** Set the rules (for deserialization)
	 *
	 * @param JSON string containing the rules
	 */
	void set_rules(const std::string& rules);

	/** Get the rules (for serialization or conversion)
	 *
	 * @return JSON string containing the rules
	 */
	std::string get_rules() const;

	/** Reset the rules
	 */
	void reset_rules();

  public:
	void setVisible(bool v) override;

  private:
	/** Execute javascript statement in a synchroneous way in the main Qt thread
	 *
	 * @param javascript the javascript content to be executed
	 * @param result the potential resulting string
	 */
	void run_javascript(const std::string& javascript, std::string* result = nullptr) const;

	/** Destroy and recreate the underling web widget
	 */
	void reinit();

	void workaround_qwebengine_refresh_bug();

  Q_SIGNALS:
	void run_javascript_signal(const QString& javascript, QString* result /*= nullptr*/) const;

  private Q_SLOTS:
	void run_javascript_slot(const QString& javascript, QString* result /*= nullptr*/) const;

  protected:
#ifdef QT_WEBKIT
	QWebView* _view;
#else
	QWebEngineView* _view;
#endif

	// This is to ensure that run_javascript signal/slot are executed in
	// a synchronous way even when called from a thread
	mutable std::mutex _mutex;
	mutable std::condition_variable _cv;
	mutable std::atomic<bool> _javascript_executed;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVQUERYBUILDER_H__
