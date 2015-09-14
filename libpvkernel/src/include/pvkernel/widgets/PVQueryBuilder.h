/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#ifndef __PVWIDGETS_PVQUERYBUILDER_H__
#define __PVWIDGETS_PVQUERYBUILDER_H__

#include <QApplication>
#ifdef QT_WEBKIT
class QWebView;
#else
class QWebEngineView;
#endif

#include <QWidget>

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
	PVQueryBuilder(QWidget* parent = nullptr);

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
	void setVisible(bool v);

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

signals:
	void run_javascript_signal(const QString& javascript, QString* result /*= nullptr*/) const;

private slots:
	void run_javascript_slot(const QString& javascript, QString* result /*= nullptr*/) const;

protected:
#ifdef QT_WEBKIT
	QWebView* 		_view;
#else
	QWebEngineView* _view;
#endif
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVQUERYBUILDER_H__
