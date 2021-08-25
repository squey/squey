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

#include <pvbase/general.h>

#include "PVInputTypeERF.h"
#include "PVERFParamsWidget.h"

bool PVRush::PVInputTypeERF::createWidget(hash_formats& formats,
                                          list_inputs& inputs,
                                          QString& format,
                                          PVCore::PVArgumentList& /*args_ext*/,
                                          QWidget* parent) const
{
	PVERFParamsWidget* params = new PVERFParamsWidget(this, parent);
	if (params->result() == QDialog::Rejected or params->exec() == QDialog::Rejected) {
		return false;
	}

	size_t source_index = 0;
	for (auto& [selected_nodes, source_name, format] : params->get_sources_info()) {
		PVRush::PVERFDescription* desc = new PVRush::PVERFDescription(
		    params->paths(), source_name.c_str(), std::move(selected_nodes));
		PVInputDescription_p ind(desc);
		inputs.push_back(ind);
		formats[QString("custom") + QString::number(source_index++)] = std::move(format);
	}
	format = "custom";

	return true;
}

QString PVRush::PVInputTypeERF::name() const
{
	return QString("erf");
}

QString PVRush::PVInputTypeERF::human_name() const
{
	return QString("ERF import plugin");
}

QString PVRush::PVInputTypeERF::human_name_serialize() const
{
	return QString("ERF");
}

QString PVRush::PVInputTypeERF::internal_name() const
{
	return QString("07-elasticsearch");
}

QString PVRush::PVInputTypeERF::menu_input_name() const
{
	return QString("ERF...");
}

QString PVRush::PVInputTypeERF::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeERF::get_custom_formats(PVInputDescription_p /*in*/,
                                                hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeERF::menu_shortcut() const
{
	return QKeySequence();
}
