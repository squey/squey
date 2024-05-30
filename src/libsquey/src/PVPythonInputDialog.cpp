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

#include <squey/PVPythonInputDialog.h>

#include <QApplication>
#include <QInputDialog>

void Squey::PVPythonInputDialog::register_functions(pysquey_t& pysquey)
{
    input_integer(pysquey);
    input_double(pysquey);
    input_text(pysquey);
    input_item(pysquey);
}

void Squey::PVPythonInputDialog::input_integer(pysquey_t& pysquey)
{
    static constexpr const char function_name[] = "input_integer";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
    auto input_integer_f = [](
        const std::string& variable_name = "integer",
        const std::string& title_name = "input integer",
        int value = 0,
        int min = -2147483647,
        int max = 2147483647,
        int step = 1)
    {
        bool ok;
        int i = value;
        QMetaObject::invokeMethod(qApp, [&]() {
            i = QInputDialog::getInt(nullptr, title_name.c_str(), (variable_name + ":").c_str() , value, min, max, step, &ok);
        }, Qt::BlockingQueuedConnection);
        if (ok) {
            return i;
        }
        // return None
    };
#pragma GCC diagnostic pop

    pysquey.def(function_name, [&](const std::string& variable_name) {
        return input_integer_f(variable_name);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name) {
        return input_integer_f(variable_name, title_name);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value) {
        return input_integer_f(variable_name, title_name, value);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min) {
        return input_integer_f(variable_name, title_name, value, min);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max) {
        return input_integer_f(variable_name, title_name, value, min, max);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max, int step) {
        return input_integer_f(variable_name, title_name, value, min, max, step);
    });
}

void Squey::PVPythonInputDialog::input_double(pysquey_t& pysquey)
{
    static constexpr const char function_name[] = "input_double";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
    auto input_double_f = [](
        const std::string& variable_name = "double",
        const std::string& title_name = "input double",
        double value = 0,
        double min = -2147483647,
        double max = 2147483647,
        int decimals = 1)
    {
        bool ok;
        double d = value;
        QMetaObject::invokeMethod(qApp, [&]() {
            d = QInputDialog::getDouble(nullptr, title_name.c_str(), (variable_name + ":").c_str() , value, min, max, decimals, &ok);
        }, Qt::BlockingQueuedConnection);
        if (ok) {
            return d;
        }
        // return None
    };
#pragma GCC diagnostic pop

    pysquey.def(function_name, [&](const std::string& variable_name) {
        return input_double_f(variable_name);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name) {
        return input_double_f(variable_name, title_name);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value) {
        return input_double_f(variable_name, title_name, value);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min) {
        return input_double_f(variable_name, title_name, value, min);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max) {
        return input_double_f(variable_name, title_name, value, min, max);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max, int decimals) {
        return input_double_f(variable_name, title_name, value, min, max, decimals);
    });
}

void Squey::PVPythonInputDialog::input_text(pysquey_t& pysquey)
{
    static constexpr const char function_name[] = "input_text";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
    auto input_text_f = [](
        const std::string& variable_name = "text",
        const std::string& title_name = "input text",
        const std::string& value = "")
    {
        bool ok;
        QString text = QString::fromStdString(value);
        QMetaObject::invokeMethod(qApp, [&]() {
            text = QInputDialog::getText(nullptr, title_name.c_str(), (variable_name + ":").c_str() , QLineEdit::Normal, QString::fromStdString(value), &ok);
        }, Qt::BlockingQueuedConnection);
        if (ok) {
            return text.toStdString();
        }
        // return None
    };
#pragma GCC diagnostic pop

    pysquey.def(function_name, [&](const std::string& variable_name) {
        return input_text_f(variable_name);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name) {
        return input_text_f(variable_name, title_name);
    });
    pysquey.def(function_name, [&](const std::string& variable_name, const std::string& title_name, const std::string& value) {
        return input_text_f(variable_name, title_name, value);
    });
}

void Squey::PVPythonInputDialog::input_item(pysquey_t& pysquey)
{
    static constexpr const char function_name[] = "input_item";

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
    auto input_item_f = [](
        const QStringList& items = {},
        int value = 0,
        const std::string& variable_name = "item",
        const std::string& title_name = "input item")
    {
        bool ok;
        QString text;
        QMetaObject::invokeMethod(qApp, [&]() {
            text = QInputDialog::getItem(nullptr, title_name.c_str(), (variable_name + ":").c_str(), items, value, true /*ediatable*/, &ok);
        }, Qt::BlockingQueuedConnection);
        if (ok) {
            return text.toStdString();
        }
        // return None
    };
#pragma GCC diagnostic pop

    auto to_qstringlist = [](const std::vector<std::string>& items) {
        QStringList result;
        for (const std::string& item : items) {
            result << QString::fromStdString(item);
        }
        return result;
    };

    pysquey.def(function_name, [&](const std::vector<std::string>& items) {
        return input_item_f(to_qstringlist(items));
    });
    pysquey.def(function_name, [&](const std::vector<std::string>& items, int value) {
        return input_item_f(to_qstringlist(items), value);
    });
    pysquey.def(function_name, [&](const std::vector<std::string>& items, int value, const std::string& variable_name) {
        return input_item_f(to_qstringlist(items), value, variable_name);
    });
    pysquey.def(function_name, [&](const std::vector<std::string>& items, int value, const std::string& variable_name, const std::string& title_name) {
        return input_item_f(to_qstringlist(items), value, variable_name, title_name);
    });
}
