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

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVAxesCombination.h>

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>
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
	         PVRush::PVFormat format);
	PVSource(Inendi::PVScene& scene,
	         PVRush::PVInputType::list_inputs_desc const& inputs,
	         PVRush::PVSourceCreator_p sc,
	         PVRush::PVFormat format,
	         size_t ext_start,
	         size_t ext_end);
	PVSource(PVScene& scene, const PVRush::PVSourceDescription& descr)
	    : PVSource(scene, descr.get_inputs(), descr.get_source_creator(), descr.get_format())
	{
	}
	~PVSource();

  public:
	void load_data(std::string const& nraw_folder)
	{
		// Try to load nraw from folder if folder looks to be a possible entry.
		// If load can't be done, do an import from input file.
		// If it is an other error, forward it, we will not handle it here.
		if (nraw_folder != "") {
			try {
				load_from_disk(nraw_folder);
				return;
			} catch (PVRush::NrawLoadingFail const& e) {
			} catch (...) {
				throw;
			}
		}

		// If the nraw can't be find, try an import from source file and format.
		// Extract the source

		PVRush::PVControllerJob_p job_import;
		job_import = extract(0);

		wait_extract_end(job_import);
	}
	/* Functions */
	PVCol get_column_count() const;

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

	PVRush::PVExtractor const& get_extractor() const { return _extractor; }

	/**
	 * Start extraction of data for current source.
	 *
	 * @param skip_lines_count : Number of line to skip at the beginning of the file.
	 *
	 * @return : Pointer to the started job.
	 */
	PVRush::PVControllerJob_p extract(size_t skip_lines_count);
	void wait_extract_end(PVRush::PVControllerJob_p job);

	void load_from_disk(std::string const& nraw_folder);

	inline PVAxesCombination& get_axes_combination() { return _axes_combination; }
	inline PVAxesCombination const& get_axes_combination() const { return _axes_combination; }

	std::map<size_t, std::string> const& get_invalid_evts() const { return _inv_elts; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	PVRush::PVSourceCreator_p get_source_creator() const { return _src_plugin; }
	std::string get_name() const
	{
		return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs).toStdString();
	}
	QString get_format_name() const { return _extractor.get_format().get_format_name(); }
	QString get_window_name() const;
	QString get_tooltip() const;

	PVView* last_active_view() const { return _last_active_view; }

	PVView* current_view();
	PVView const* current_view() const;

	PVRush::PVFormat& get_format() { return _extractor.get_format(); }
	PVRush::PVFormat const& get_format() const { return _extractor.get_format(); }

	virtual std::string get_serialize_description() const { return "Source: " + get_name(); }

	size_t get_extraction_last_nlines() const { return _extractor.get_last_nlines(); }
	size_t get_extraction_last_start() const { return _extractor.get_last_start(); }

  public:
	virtual QString get_children_description() const { return "Mapped(s)"; }
	virtual QString get_children_serialize_name() const { return "mapped"; }

	inline void set_last_active_view(Inendi::PVView* view) { _last_active_view = view; }

  public:
	static PVSource& serialize_read(PVCore::PVSerializeObject& so, PVScene& parent);
	void serialize_write(PVCore::PVSerializeObject& so);

  private:
	void files_append_noextract();
	void extract_finished();

  private:
	PVView* _last_active_view = nullptr;

	PVRush::PVExtractor _extractor; //!< Tool to extract data and generate NRaw.
	PVRush::PVInputType::list_inputs _inputs;

	PVRush::PVSourceCreator_p _src_plugin;
	PVRush::PVNraw& _nraw;                   //!< Reference to Nraw data (owned by extractor)
	std::map<size_t, std::string> _inv_elts; //!< List of invalid elements sorted by line number.

	PVAxesCombination _axes_combination;
};
}

#endif /* INENDI_PVSOURCE_H */
