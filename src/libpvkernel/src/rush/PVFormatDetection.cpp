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

#include <pvkernel/rush/PVFormatDetection.h>

#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVTypesDiscoveryOutput.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>

#include <QObject>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <tuple>

QDomDocument PVRush::format_of_splitter(PVFilter::PVFieldsSplitter const& splitter,
                                        PVCol axes_count,
                                        size_t preamble_lines)
{
	QDomDocument document;
	// It writes the header of the document, which is all that its node is for here.
	const std::unique_ptr<PVXmlTreeNodeDom> root(PVXmlTreeNodeDom::new_format(document));
	QDomElement root_element = document.documentElement();
	root_element.setAttribute("first_line", qulonglong(preamble_lines));

	QDomElement splitter_element = document.createElement(splitter.type_name());
	splitter_element.setAttribute("type", splitter.registered_name());
	PVXmlTreeNodeDom::setFromArgumentList(splitter_element, splitter.get_default_args(),
	                                      splitter.get_args());
	root_element.appendChild(splitter_element);
	for (PVCol axis_index(0); axis_index < axes_count; axis_index++) {
		QDomElement field = document.createElement(PVFORMAT_XML_TAG_FIELD_STR);
		QDomElement axis = document.createElement(PVFORMAT_XML_TAG_AXIS_STR);
		axis.setAttribute(PVFORMAT_AXIS_NAME_STR, QString("Axis %1").arg(axis_index + 1));
		axis.setAttribute(PVFORMAT_AXIS_TITLECOLOR_STR, PVFORMAT_AXIS_TITLECOLOR_DEFAULT);
		axis.setAttribute(PVFORMAT_AXIS_COLOR_STR, PVFORMAT_AXIS_COLOR_DEFAULT);
		axis.setAttribute(PVFORMAT_AXIS_MAPPING_STR, PVFORMAT_AXIS_MAPPING_DEFAULT);
		axis.setAttribute(PVFORMAT_AXIS_SCALING_STR, PVFORMAT_AXIS_SCALING_DEFAULT);
		axis.setAttribute(PVFORMAT_AXIS_TYPE_STR, PVFORMAT_AXIS_TYPE_DEFAULT);
		field.appendChild(axis);
		splitter_element.appendChild(field);
		PVXmlTreeNodeDom(axis).setMappingProperties("string", {}, {});
	}
	return document;
}

PVRow PVRush::types_discovery_row_count(PVCol axes_count)
{
	static constexpr size_t total_fields = 100000;
	static constexpr size_t multiple_to_round = 100;
	return PVRow(((total_fields / size_t(axes_count) + multiple_to_round - 1) / multiple_to_round) *
	             multiple_to_round);
}

bool PVRush::set_discovered_types(QDomElement format_root,
                                  PVTypesDiscoveryOutput const& discovered,
                                  bool name_axes)
{
	const QDomNodeList axes = format_root.elementsByTagName(PVFORMAT_XML_TAG_AXIS_STR);
	bool has_header = false;
	for (PVCol axis_index(0); axis_index < axes.size(); axis_index++) {
		QDomElement axis = axes.at(axis_index).toElement();
		const auto [type, type_format, name] = discovered.type_desc(axis_index);
		axis.setAttribute(PVFORMAT_AXIS_TYPE_STR, type.c_str());
		axis.setAttribute("type_format", type_format.c_str());
		if (not name.empty() and name_axes) {
			axis.setAttribute(PVFORMAT_AXIS_NAME_STR, name.c_str());
			has_header = true;
		}
	}
	return has_header;
}

PVRush::PVFormat PVRush::detect_format(PVInputDescription_p const& input,
                                       PVSourceCreator_p const& creator,
                                       PVCore::PVTaskProgress* progress)
{
	PVCol axes_count;
	size_t preamble_lines = 0;
	const PVFilter::PVFieldsSplitter_p splitter =
	    PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(
	        creator->create_source_from_input(input), axes_count, &preamble_lines);
	if (not splitter) {
		throw PVFormatInvalid(
		    QObject::tr("Squey found no splitter that cuts the lines of %1 into fields.")
		        .arg(input->human_name())
		        .toStdString());
	}
	QDomDocument document = format_of_splitter(*splitter, axes_count, preamble_lines);

	PVTypesDiscoveryOutput discovered;
	PVExtractor extractor(PVFormat(document.documentElement()), discovered, creator,
	                      PVInputType::list_inputs{input});
	const PVRow row_count = types_discovery_row_count(axes_count);
	if (progress != nullptr) {
		progress->set_maximum(row_count);
	}
	const PVControllerJob_p job = extractor.process_from_agg_idxes(0, row_count - 1);
	while (job->running()) {
		if (progress != nullptr) {
			if (progress->canceled()) {
				job->cancel();
				break;
			}
			progress->set_value(job->status());
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	job->wait_end();
	if (progress != nullptr and progress->canceled()) {
		return {};
	}

	QDomElement root = document.documentElement();
	if (set_discovered_types(root, discovered, true)) {
		root.setAttribute("first_line", qulonglong(preamble_lines + 1));
	}
	return PVFormat(root);
}
