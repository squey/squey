/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#ifndef __PVWIDGETS_PVQUERYBUILDER_H__
#define __PVWIDGETS_PVQUERYBUILDER_H__

#include <QApplication>
#include <QtWebEngineWidgets/QWebEngineView>

namespace PVWidgets
{

class PVQueryBuilder : public QWidget
{
	Q_OBJECT;

private:
	using columns_t = std::vector<std::pair<std::string, std::string>>;

public:
	PVQueryBuilder(QWidget* parent = nullptr);

public:
	void set_filters(const std::string& filters);
	void set_filters(const columns_t& cols);

public:
	void set_rules(const std::string& rules);
	std::string get_rules() const;
	void reset();

public:
	void setVisible(bool v);

private:
	void run_javascript(const std::string& javascript, std::string* result = nullptr) const;
	void reinit();

	void workaround_qwebengine_refresh_bug();
	bool workaround_qwebengine_refresh_bug_toggle = false;

protected:
	QWebEngineView* _view;
};

} // namespace PVWidgets

#endif // __PVWIDGETS_PVQUERYBUILDER_H__
