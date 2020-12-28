/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#include <inendi/PVPythonInputDialog.h>

#include <QApplication>
#include <QInputDialog>

void Inendi::PVPythonInputDialog::register_functions(inspyctor_t& inspyctor)
{
    input_integer(inspyctor);
    input_double(inspyctor);
    input_text(inspyctor);
    input_item(inspyctor);
}

void Inendi::PVPythonInputDialog::input_integer(inspyctor_t& inspyctor)
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

    inspyctor.def(function_name, [&](const std::string& variable_name) {
        return input_integer_f(variable_name);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name) {
        return input_integer_f(variable_name, title_name);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value) {
        return input_integer_f(variable_name, title_name, value);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min) {
        return input_integer_f(variable_name, title_name, value, min);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max) {
        return input_integer_f(variable_name, title_name, value, min, max);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max, int step) {
        return input_integer_f(variable_name, title_name, value, min, max, step);
    });
}

void Inendi::PVPythonInputDialog::input_double(inspyctor_t& inspyctor)
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

    inspyctor.def(function_name, [&](const std::string& variable_name) {
        return input_double_f(variable_name);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name) {
        return input_double_f(variable_name, title_name);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value) {
        return input_double_f(variable_name, title_name, value);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min) {
        return input_double_f(variable_name, title_name, value, min);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max) {
        return input_double_f(variable_name, title_name, value, min, max);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, int value, int min, int max, int decimals) {
        return input_double_f(variable_name, title_name, value, min, max, decimals);
    });
}

void Inendi::PVPythonInputDialog::input_text(inspyctor_t& inspyctor)
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

    inspyctor.def(function_name, [&](const std::string& variable_name) {
        return input_text_f(variable_name);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name) {
        return input_text_f(variable_name, title_name);
    });
    inspyctor.def(function_name, [&](const std::string& variable_name, const std::string& title_name, const std::string& value) {
        return input_text_f(variable_name, title_name, value);
    });
}

void Inendi::PVPythonInputDialog::input_item(inspyctor_t& inspyctor)
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

    inspyctor.def(function_name, [&](const std::vector<std::string>& items) {
        return input_item_f(to_qstringlist(items));
    });
    inspyctor.def(function_name, [&](const std::vector<std::string>& items, int value) {
        return input_item_f(to_qstringlist(items), value);
    });
    inspyctor.def(function_name, [&](const std::vector<std::string>& items, int value, const std::string& variable_name) {
        return input_item_f(to_qstringlist(items), value, variable_name);
    });
    inspyctor.def(function_name, [&](const std::vector<std::string>& items, int value, const std::string& variable_name, const std::string& title_name) {
        return input_item_f(to_qstringlist(items), value, variable_name, title_name);
    });
}