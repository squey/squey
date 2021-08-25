/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVSPLUNKPARAMSWIDGET_H
#define PVSPLUNKPARAMSWIDGET_H

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/widgets/PVPresetsWidget.h>

#include "../common/PVParamsWidget.h"
#include "../../common/splunk/PVSplunkInfos.h"
#include "../../common/splunk/PVSplunkQuery.h"
#include "PVSplunkPresets.h"

namespace PVRush
{

class PVInputTypeSplunk;

/**
 * This class represent the widget associated with the splunk input plugin.
 * It derives from PVParamsWidget : read its documentation for more information.
 */
class PVSplunkParamsWidget
    : public PVParamsWidget<PVInputTypeSplunk, PVSplunkPresets, PVSplunkInfos, PVSplunkQuery>
{
	Q_OBJECT

  private:
	enum EQueryType {
		QUERY_BUILDER = 0,
		SPLUNK,

		COUNT
	};

  public:
	PVSplunkParamsWidget(PVInputTypeSplunk const* in_t,
	                     PVRush::hash_formats const& formats,
	                     QWidget* parent);

  public:
	QString get_server_query(std::string* error = nullptr) const override;
	QString get_serialize_query() const override;

  protected Q_SLOTS:
	size_t query_result_count(std::string* error = nullptr) override;
	bool fetch_server_data(const PVSplunkInfos& infos) override;
	void query_type_changed_slot() override;

  protected:
	PVSplunkInfos get_infos() const override;
	bool set_infos(PVSplunkInfos const& infos) override;
	void set_query(QString const& query) override;
	bool check_connection(std::string* error = nullptr) override;
	void export_query_result(PVCore::PVStreamingCompressor& compressor,
	                         const std::string& sep,
	                         const std::string& quote,
	                         bool header,
	                         PVCore::PVProgressBox& pbox,
	                         std::string* error = nullptr) override;
	void set_query_type(QString const& query_type);

  private Q_SLOTS:
	void splunk_filter_changed_by_user_slot();

  private:
	QComboBox* _combo_index;
	QComboBox* _combo_host;
	QComboBox* _combo_sourcetype;
};

} // namespace PVRush

#endif
