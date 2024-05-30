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

#include "PVInputTypeElasticsearch.h"

#include "PVElasticsearchParamsWidget.h"

#include "../../common/elasticsearch/PVElasticsearchInfos.h"

bool PVRush::PVInputTypeElasticsearch::createWidget(hash_formats& formats,
                                                    list_inputs& inputs,
                                                    QString& format,
                                                    PVCore::PVArgumentList& /*args_ext*/,
                                                    QWidget* parent) const
{
	connect_parent(parent);
	std::unique_ptr<PVElasticsearchParamsWidget> params(
	    new PVElasticsearchParamsWidget(this, formats, parent));
	if (params->exec() == QDialog::Rejected) {
		return false;
	}

	auto* query = new PVElasticsearchQuery(params->get_query());

	PVInputDescription_p ind(query);
	inputs.push_back(ind);

	if (params->is_format_custom()) {
		PVRush::PVFormat custom_format(params->get_custom_format().documentElement());
		formats["custom"] = std::move(custom_format);
		format = "custom";
	} else {
		format = params->get_format_path();
	}

	return true;
}

QString PVRush::PVInputTypeElasticsearch::name() const
{
	return {"elasticsearch"};
}

QString PVRush::PVInputTypeElasticsearch::human_name() const
{
	return {"Elasticsearch import plugin"};
}

QString PVRush::PVInputTypeElasticsearch::human_name_serialize() const
{
	return {"Elasticsearch"};
}

QString PVRush::PVInputTypeElasticsearch::internal_name() const
{
	return {"03-elasticsearch"};
}

QString PVRush::PVInputTypeElasticsearch::menu_input_name() const
{
	return {"Elasticsearch"};
}

QString PVRush::PVInputTypeElasticsearch::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeElasticsearch::get_custom_formats(PVInputDescription_p /*in*/,
                                                          hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeElasticsearch::menu_shortcut() const
{
	return {};
}
