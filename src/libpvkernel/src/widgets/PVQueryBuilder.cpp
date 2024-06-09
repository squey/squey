//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVQueryBuilder.h>
#include <QtCore/qobjectdefs.h>
#include <qcolor.h>
#include <qlayout.h>
#include <qnamespace.h>
#include <qpalette.h>
#include <qurl.h>
#include <qvariant.h>
#include <qwebenginepage.h>
#include <rapidjson/allocators.h>
#include <rapidjson/encodings.h>
#include <rapidjson/rapidjson.h>
#include <stdlib.h>

#ifdef QT_WEBKIT
#include <QtWebKitWidgets/QWebView>
#include <QtWebKitWidgets/QWebFrame>
#else
#include <QtWebEngineWidgets/QWebEngineView>
#endif

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <QEventLoop>
#include <QHBoxLayout>
#include <sstream>
#include <QApplication>
#include <functional>

PVWidgets::PVQueryBuilder::PVQueryBuilder(QWidget* parent /*= nullptr*/)
    : QWidget(parent), _view(new QWebEngineView)
{
	// Javascript content must be executed on the main Qt thread to avoid crashs
	connect(this, &PVQueryBuilder::run_javascript_signal, this,
	        &PVQueryBuilder::run_javascript_slot);

	reinit();
}

static std::string get_query_builder_path()
{
	const char* path = getenv("QUERY_BUILDER_PATH");
	if (path) {
		return path;
	}
	return SQUEY_QUERY_BUILDER;
}

/**
 * Changing filters imply to destroy and recreate the whole widget
 */
void PVWidgets::PVQueryBuilder::reinit()
{
	_view->setContextMenuPolicy(Qt::NoContextMenu);

	_view->load(QUrl(std::string("file://" + get_query_builder_path() + "/index.html").c_str()));

	// Trick to wait for the page to be properly loaded
	QEventLoop loop;
	connect(_view, &QWebEngineView::loadFinished, &loop, &QEventLoop::quit);
	loop.exec();

	if (layout() == nullptr) {
		auto layout = new QHBoxLayout;
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
	// Show hourglass during loading
	QApplication::setOverrideCursor(Qt::WaitCursor);

	reinit();

	rapidjson::Document json;
	json.Parse<0>(filters.c_str());
	rapidjson::Document::AllocatorType& allocator = json.GetAllocator();

	// plugins
	rapidjson::Value json_plugins(rapidjson::kArrayType);
	std::vector<std::string> plugins = {
	    "bt-tooltip-errors",
	    //"sortable", // doesn't seem to be supported by Qt for the moment
	    "filter-description", "unique-filter", "bt-tooltip-errors", "bt-selectpicker",
	    "bt-checkbox"};
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

	// Back to normal cursor
	QApplication::restoreOverrideCursor();
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
	run_javascript("$('#querybuilder').queryBuilder('setRules', " + rules + ");");
}

void PVWidgets::PVQueryBuilder::reset_rules()
{
	run_javascript("$('#querybuilder').queryBuilder('reset');");

	workaround_qwebengine_refresh_bug();
}

std::string PVWidgets::PVQueryBuilder::get_rules() const
{
	std::string result;

	run_javascript("var obj = document.activeElement;"
	               "if (obj) {"
	               "	var event = new Event('change');"
	               "	obj.dispatchEvent(event);"
	               "}"
	               "var result = $('#querybuilder').queryBuilder('getRules');"
	               "if (!$.isEmptyObject(result)) {"
	               "	JSON.stringify(result, null, 2);"
	               "}",
	               &result);

	return result;
}

void PVWidgets::PVQueryBuilder::run_javascript(const std::string& javascript,
                                               std::string* result /*= nullptr*/) const
{
	QString r;

	_javascript_executed = false;

	Q_EMIT run_javascript_signal(javascript.c_str(), &r);

	// Yet another new trick to run asynchronous code synchronously
	if (not _javascript_executed) {
		std::unique_lock<std::mutex> lck(_mutex);
		_cv.wait(lck, [&]() { return _javascript_executed == true; });
	}

	if (result) {
		*result = r.toStdString();
	}
}

void PVWidgets::PVQueryBuilder::run_javascript_slot(const QString& javascript,
                                                    QString* result /*= nullptr*/) const
{
	QVariant r;

	QEventLoop loop;

	_view->page()->runJavaScript(javascript, [&](const QVariant& res) {
		r = res;
		Q_EMIT loop.quit();
	});

	loop.exec(); // Trick to run asynchronous code synchronously

	if (result) {
		*result = r.toString();
	}

	_javascript_executed = true;
	_cv.notify_one();
}

void PVWidgets::PVQueryBuilder::setVisible(bool v)
{
	QWidget::setVisible(v);

	workaround_qwebengine_refresh_bug();
}

void PVWidgets::PVQueryBuilder::workaround_qwebengine_refresh_bug()
{
	// Really really really ugly hack to workaround QWebEngine refresh bug
	if (_view) {
		_view->resize(_view->width() + 1, _view->height() + 1);
		_view->resize(_view->width() - 1, _view->height() - 1);
	}
}
