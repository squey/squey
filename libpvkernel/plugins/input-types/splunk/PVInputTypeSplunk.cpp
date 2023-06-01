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

#include "PVInputTypeSplunk.h"

#include "PVSplunkParamsWidget.h"

#include "../../common/splunk/PVSplunkInfos.h"

bool PVRush::PVInputTypeSplunk::createWidget(hash_formats& formats,
                                             list_inputs& inputs,
                                             QString& format,
                                             PVCore::PVArgumentList& /*args_ext*/,
                                             QWidget* parent) const
{
	connect_parent(parent);
	std::unique_ptr<PVSplunkParamsWidget> params(new PVSplunkParamsWidget(this, formats, parent));
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	auto* query = new PVSplunkQuery(params->get_query());

	PVInputDescription_p ind(query);
	inputs.push_back(ind);

	format = SQUEY_BROWSE_FORMAT_STR;
	const QString& format_path = params->get_format_path();
	if (format_path.isEmpty()) {
		format = SQUEY_BROWSE_FORMAT_STR;
	} else {
		format = format_path;
	}

	return true;
}

QString PVRush::PVInputTypeSplunk::name() const
{
	return {"splunk"};
}

QString PVRush::PVInputTypeSplunk::human_name() const
{
	return {"Splunk import plugin"};
}

QString PVRush::PVInputTypeSplunk::human_name_serialize() const
{
	return {"Splunk"};
}

QString PVRush::PVInputTypeSplunk::internal_name() const
{
	return {"05-splunk"};
}

QString PVRush::PVInputTypeSplunk::menu_input_name() const
{
	return {"Splunk..."};
}

QString PVRush::PVInputTypeSplunk::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeSplunk::get_custom_formats(PVInputDescription_p /*in*/,
                                                   hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeSplunk::menu_shortcut() const
{
	return {};
}
