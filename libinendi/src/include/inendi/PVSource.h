/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSOURCE_H
#define INENDI_PVSOURCE_H

#include <QString>
#include <QList>
#include <QStringList>
#include <QVector>

#include <pvkernel/core/PVDataTreeObject.h>
#include <pvkernel/core/PVSerializeArchive.h>

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVSourceDescription.h>

#include <inendi/PVMapped.h>

namespace Inendi
{

class PVScene;

/**
 * \class PVSource
 */
class PVSource : public PVCore::PVDataTreeParent<PVMapped, PVSource>,
                 public PVCore::PVDataTreeChild<PVScene, PVSource>
{
  public:
	PVSource(Inendi::PVScene& scene,
	         PVRush::PVInputType::list_inputs_desc const& inputs,
	         PVRush::PVSourceCreator_p sc,
	         PVRush::PVFormat const& format);
	PVSource(PVScene& scene, const PVRush::PVSourceDescription& descr)
	    : PVSource(scene, descr.get_inputs(), descr.get_source_creator(), descr.get_format())
	{
	}
	~PVSource();

  public:
	void load_data() { wait_extract_end(extract(0)); }

	/* Functions */
	PVCol get_nraw_column_count() const;

	PVRush::PVNraw& get_rushnraw();
	const PVRush::PVNraw& get_rushnraw() const;

	std::string get_value(PVRow row, PVCol col) const;

	/**
	 * Get the original value encountered during import, even if it
	 * has failed to be converted correctly
	 */
	std::string get_input_value(PVRow row, PVCol col, bool* failed = nullptr) const;

	/**
	 * Check if the value encountered during import has failed
	 * to be converted correctly
	 *
	 * @return true if the conversion has failed
	 *         false otherwise
	 */
	bool has_conversion_failed(PVRow row, PVCol col) const;

	/**
	 * Return the number of row in the datastorage.
	 */
	PVRow get_row_count() const;

	/**
	 * Return number of correctly splitted row in the datastorage.
	 */
	PVRow get_valid_row_count() const;

	/**
	 * Start extraction of data for current source.
	 *
	 * @param skip_lines_count : Number of line to skip at the beginning of the file.
	 *
	 * @return : Pointer to the started job.
	 */
	PVRush::PVControllerJob_p extract(size_t skip_lines_count);
	void wait_extract_end(PVRush::PVControllerJob_p job);

	std::map<size_t, std::string> const& get_invalid_evts() const { return _inv_elts; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	PVRush::PVSourceCreator_p get_source_creator() const { return _src_plugin; }
	std::string get_name() const
	{
		return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs).toStdString();
	}
	QString get_format_name() const { return _format.get_format_name(); }
	QString get_window_name() const;
	QString get_tooltip() const;

	PVView* last_active_view() const { return _last_active_view; }

	PVView* current_view();
	PVView const* current_view() const;

	PVRush::PVFormat const& get_format() const { return _format; }

	virtual std::string get_serialize_description() const { return "Source: " + get_name(); }

  public:
	virtual QString get_children_description() const { return "Mapped(s)"; }
	virtual QString get_children_serialize_name() const { return "mapped"; }

	inline void set_last_active_view(Inendi::PVView* view) { _last_active_view = view; }

  public:
	static PVSource& serialize_read(PVCore::PVSerializeObject& so, PVScene& parent);
	void serialize_write(PVCore::PVSerializeObject& so);

  private:
	PVView* _last_active_view = nullptr;

	PVRush::PVFormat
	    _format;          //!< Format use to create the source (also contains metadata like colors)
	PVRush::PVNraw _nraw; //!< NRaw data
	PVRush::PVInputType::list_inputs _inputs;

	PVRush::PVSourceCreator_p _src_plugin;
	// FIXME : The extracor is an attribute as we can't create an extractor from source (it would
	// required a move constructor)
	PVRush::PVExtractor _extractor;
	std::map<size_t, std::string> _inv_elts; //!< List of invalid elements sorted by line number.
};
}

#endif /* INENDI_PVSOURCE_H */
