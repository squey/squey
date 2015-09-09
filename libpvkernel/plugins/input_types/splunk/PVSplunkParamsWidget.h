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

class PVSplunkParamsWidget: public PVParamsWidget<PVInputTypeSplunk, PVSplunkPresets, PVSplunkInfos, PVSplunkQuery>
{
	Q_OBJECT

public:
	PVSplunkParamsWidget(PVInputTypeSplunk const* in_t, PVRush::hash_formats const& formats, QWidget* parent);

public:
	QString get_server_query(std::string* error = nullptr) const override;
	QString get_serialize_query() const override;

protected slots:
	size_t query_result_count(std::string* error = nullptr) override;
	bool fetch_server_data(const PVSplunkInfos& infos) override;
	void query_type_changed_slot() override;

protected:
	PVSplunkInfos get_infos() const override;
	bool set_infos(PVSplunkInfos const& infos);
	void set_query(QString const& query);
	bool check_connection(std::string* error = nullptr) override;
	void export_query_result(QTextStream& output_stream, PVCore::PVProgressBox& pbox, std::string* error = nullptr) override;
	void set_query_type(QString const& query_type);

private slots:
	void splunk_filter_changed_by_user_slot();

private:
	QComboBox* _combo_index;
	QComboBox* _combo_host;
	QComboBox* _combo_sourcetype;
};

} // namespace PVRush

#endif
