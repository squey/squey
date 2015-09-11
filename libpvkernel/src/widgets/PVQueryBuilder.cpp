/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#include <pvkernel/widgets/PVQueryBuilder.h>

#ifdef QT_WEBKIT
#include <QtWebKitWidgets/QWebView>
#include <QtWebKitWidgets/QWebFrame>
#else
#include <QtWebEngineWidgets/QWebEngineView>
#endif

#include <QDir>
#include <QEventLoop>
#include <QInputDialog>
#include <QHBoxLayout>
#include <QTextStream>
#include <QMessageBox>

#include <sstream>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

PVWidgets::PVQueryBuilder::PVQueryBuilder(QWidget* parent /*= nullptr*/) : QWidget(parent),
#ifdef QT_WEBKIT
	_view(new QWebView)
#else
	_view(new QWebEngineView)
#endif
{
	reinit();
}

/**
 * Changing filters imply to destroy and recreate the whole widget
 */
void PVWidgets::PVQueryBuilder::reinit()
{
	_view->setContextMenuPolicy(Qt::NoContextMenu);

	const char* querybuilder_dir = std::getenv("PICVIZ_QUERYBUILDER_DIR");
	assert((querybuilder_dir != nullptr) && "PICVIZ_QUERYBUILDER_DIR not set");
	_view->load(QUrl(std::string("file://" + std::string(querybuilder_dir) + "/index.html").c_str()));

	// Trick to wait for the page to be properly loaded
	QEventLoop loop;
	connect(_view, SIGNAL(loadFinished(bool)), &loop, SLOT(quit()));
	loop.exec();

	if (layout() == nullptr) {
		QHBoxLayout* layout = new QHBoxLayout;
		setLayout(layout);
	}
	layout()->addWidget(_view);

	// Set proper color
	QColor bg_color = parentWidget()->palette().color(QWidget::backgroundRole()).lighter(110);
	std::stringstream js_color;
	js_color << "document.body.style.background = \"" << qPrintable(bg_color.name()) << "\"";
	run_javascript(js_color.str().c_str());

	workaround_qwebengine_refresh_bug();
}

void PVWidgets::PVQueryBuilder::set_filters(const std::string& filters)
{
	reinit();

	rapidjson::Document json;
	json.Parse<0>(filters.c_str());
	rapidjson::Document::AllocatorType& allocator = json.GetAllocator();

	// plugins
	rapidjson::Value json_plugins(rapidjson::kArrayType);
	std::vector<std::string> plugins = {
		"bt-tooltip-errors",
		//"sortable", // doesn't seem to be supported by Qt for the moment
		"filter-description",
		"unique-filter",
		"bt-tooltip-errors",
		"bt-selectpicker",
		"bt-checkbox"
	};
	for (const std::string& plugin : plugins) {
		rapidjson::Value json_plugin;
		json_plugin.SetString(plugin.c_str(), json.GetAllocator());

		json_plugins.PushBack(json_plugin, allocator);
	}
	json.AddMember("plugins", json_plugins, allocator);

	rapidjson::StringBuffer strbuf;
	rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
	json.Accept(writer);

	std::stringstream js;

	js << "$('#querybuilder').queryBuilder(" << strbuf.GetString() << ");";

	run_javascript(js.str().c_str());
}

void PVWidgets::PVQueryBuilder::set_filters(const columns_t& columns)
{
	rapidjson::Document json;
	rapidjson::Document::AllocatorType& allocator = json.GetAllocator();
	json.SetObject();

	rapidjson::Value json_filters(rapidjson::kArrayType);
	for (const auto& column : columns) {
		const std::string& name = column.first;
		const std::string& type = column.second;

		rapidjson::Value val;
		rapidjson::Value obj;
		obj.SetObject();

		val.SetString(name.c_str(), json.GetAllocator());
		obj.AddMember("name", val, allocator);

		val.SetString(name.c_str(), json.GetAllocator());
		obj.AddMember("id", val, allocator);

		val.SetString(type.c_str(), json.GetAllocator());
		obj.AddMember("type", val, allocator);

		json_filters.PushBack(obj, allocator);
	}
	json.AddMember("filters", json_filters, allocator);

	rapidjson::StringBuffer strbuf;
	rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
	json.Accept(writer);

	set_filters(strbuf.GetString());
}

void PVWidgets::PVQueryBuilder::set_rules(const std::string& rules)
{
	run_javascript(
		"$('#querybuilder').queryBuilder('setRules', " + rules + ");"
	);
}

void PVWidgets::PVQueryBuilder::reset_rules()
{
	run_javascript(
		"$('#querybuilder').queryBuilder('reset');"
	);

	workaround_qwebengine_refresh_bug();
}

std::string PVWidgets::PVQueryBuilder::get_rules() const
{
	std::string result;

	run_javascript(
		"var result = $('#querybuilder').queryBuilder('getRules');"
		"if (!$.isEmptyObject(result)) {"
		"	JSON.stringify(result, null, 2);"
		"}",
		&result
	);

	return result;
}

void PVWidgets::PVQueryBuilder::run_javascript(const std::string& javascript, std::string* result /*= nullptr*/) const
{
	QVariant r;

#ifdef QT_WEBKIT
	r = _view->page()->mainFrame()->evaluateJavaScript(javascript.c_str());
#else
	QEventLoop loop;

	_view->page()->runJavaScript(javascript.c_str(), [&](const QVariant& res)
		{
			r = res;
			emit loop.quit();
		}
	);

	loop.exec(); // Trick to run asynchronous code synchronously
#endif

	if (result) {
		*result = r.toString().toStdString();
	}

}

void PVWidgets::PVQueryBuilder::setVisible(bool v)
{
	QWidget::setVisible(v);

	workaround_qwebengine_refresh_bug();
}

void PVWidgets::PVQueryBuilder::workaround_qwebengine_refresh_bug()
{
#ifndef QT_WEBKIT
	// Really really really ugly hack to workaround QWebEngine refresh bug
	if (_view) {
		_view->resize(_view->width() +1, _view->height() +1);
		_view->resize(_view->width() -1, _view->height() -1);
	}
#endif
}
