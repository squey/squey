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

#include "PVInputTypeDatabase.h"
#include "PVDatabaseParamsWidget.h"

#include "../../common/database/PVDBInfos.h"

#include <pvkernel/widgets/PVDialogImportWidget.h>

#include <QDialogButtonBox>

PVRush::PVInputTypeDatabase::PVInputTypeDatabase() : PVInputTypeDesc<PVDBQuery>() {}

bool PVRush::PVInputTypeDatabase::create_widget(hash_formats& formats,
                                               list_inputs& inputs,
                                               QString& format,
                                               PVCore::PVArgumentList& /*args_ext*/,
                                               QWidget* parent) const
{
	connect_parent(parent);
	auto* params = new PVDatabaseParamsWidget(this, formats, parent);
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	PVDBInfos infos = params->get_infos();
	PVDBServ_p serv(new PVDBServ(infos));
	PVInputDescription_p query(new PVDBQuery(serv, params->get_query()));

	inputs.push_back(query);

	if (params->is_format_custom()) {
		PVRush::PVFormat custom_format(params->get_custom_format().documentElement());
		formats["custom"] = custom_format;
		format = "custom";
	} else {
		format = params->get_existing_format();
	}

	return true;
}

PVWidgets::PVImportWidget* PVRush::PVInputTypeDatabase::create_import_widget(QWidget* parent) const
{
	auto* params = new PVDatabaseParamsWidget(this, hash_formats(), nullptr);
	auto* buttons = params->findChild<QDialogButtonBox*>("buttonBox");
	const auto sources = [params]() -> std::vector<PVWidgets::PVImportWidget::source_t> {
		const PVInputDescription_p query(
		    new PVDBQuery(PVDBServ_p(new PVDBServ(params->get_infos())), params->get_query()));
		return {{{query},
		         params->is_format_custom()
		             ? PVFormat(params->get_custom_format().documentElement())
		             : PVWidgets::PVDialogImportWidget::format_of(params->get_existing_format())}};
	};
	return new PVWidgets::PVDialogImportWidget(params,
	                                           {.accept = buttons->button(QDialogButtonBox::Open),
	                                            .buttons = {buttons},
	                                            .sources = sources},
	                                           parent);
}

PVRush::PVInputTypeDatabase::~PVInputTypeDatabase() = default;

QString PVRush::PVInputTypeDatabase::name() const
{
	return {"database"};
}

QString PVRush::PVInputTypeDatabase::human_name() const
{
	return {"SQL databases import plugin"};
}

QString PVRush::PVInputTypeDatabase::human_name_serialize() const
{
	return {"Databases"};
}

QString PVRush::PVInputTypeDatabase::internal_name() const
{
	return {"02-database"};
}

QString PVRush::PVInputTypeDatabase::menu_input_name() const
{
	return {"SQL"};
}

QString PVRush::PVInputTypeDatabase::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeDatabase::get_custom_formats(PVInputDescription_p /*in*/,
                                                     hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeDatabase::menu_shortcut() const
{
	return {};
}
