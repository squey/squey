//
// MIT License
//
// © Squey, 2026
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

#include <pvkernel/core/PVArgumentSettings.h>

#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVSettings.h>

#include <optional>

namespace
{

/** The setting that holds \p argument, without its key, title and description. */
std::optional<PVCore::PVSettingDescriptor> setting_of(PVCore::PVArgument const& argument)
{
	PVCore::PVSettingDescriptor setting{.key = {},
	                                    .type = PVCore::PVSettingType::String,
	                                    .default_value = argument,
	                                    .title = {},
	                                    .description = {}};
	switch (argument.typeId()) {
	case QMetaType::Bool:
		setting.type = PVCore::PVSettingType::Bool;
		return setting;
	case QMetaType::Int:
	case QMetaType::LongLong:
		setting.type = PVCore::PVSettingType::Int;
		return setting;
	case QMetaType::UInt:
	case QMetaType::ULongLong:
		setting.type = PVCore::PVSettingType::Int;
		setting.minimum = 0;
		return setting;
	case QMetaType::Double:
		setting.type = PVCore::PVSettingType::Double;
		return setting;
	case QMetaType::QString:
		return setting;
	case QMetaType::QChar:
		// No character, as a disabled CSV escape, is the empty string.
		setting.default_value = argument.toChar().isNull() ? QString() : QString(argument.toChar());
		setting.max_length = 1;
		return setting;
	default:
		break;
	}
	if (argument.userType() == qMetaTypeId<PVCore::PVEnumType>()) {
		const auto enumeration = argument.value<PVCore::PVEnumType>();
		setting.type = PVCore::PVSettingType::Choice;
		for (QString const& label : enumeration.get_list()) {
			setting.choices.push_back({label, label});
		}
		setting.default_value = enumeration.get_sel();
		return setting;
	}
	return std::nullopt;
}

} // namespace

PVCore::PVArgumentKeyList PVCore::declare_argument_settings(QString const& section,
                                                            PVArgumentList const& args,
                                                            PVArgumentKeyList const& keys,
                                                            QString const& description)
{
	PVArgumentKeyList declared;
	for (PVArgumentKey const& key : keys) {
		auto argument = args.find(key);
		if (argument == args.end()) {
			continue;
		}
		std::optional<PVSettingDescriptor> setting = setting_of(argument->value());
		if (not setting) {
			continue;
		}
		setting->key = section + "." + key.key();
		setting->title = argument->key().desc();
		setting->description = description;
		PVSettings::get().declare(*setting);
		declared.push_back(key);
	}
	return declared;
}

void PVCore::apply_argument_settings(QString const& section, PVArgumentList& args)
{
	PVSettings& settings = PVSettings::get();
	for (auto& argument : args) {
		const QVariant value = settings.value(section + "." + argument.key().key());
		if (not value.isValid()) {
			continue;
		}
		PVArgument& held = argument.value();
		if (held.userType() == qMetaTypeId<PVEnumType>()) {
			auto enumeration = held.value<PVEnumType>();
			const auto index = enumeration.get_list().indexOf(value.toString());
			if (index >= 0) {
				enumeration.set_sel(int(index));
				held.setValue(enumeration);
			}
		} else if (held.typeId() == QMetaType::QChar) {
			const QString text = value.toString();
			held = text.isEmpty() ? QChar() : text.front();
		} else {
			QVariant converted = value;
			if (converted.convert(held.metaType())) {
				held = converted;
			}
		}
	}
}
