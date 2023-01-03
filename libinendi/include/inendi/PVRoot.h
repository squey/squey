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

#ifndef INENDI_PVROOT_H
#define INENDI_PVROOT_H

#include <inendi/PVCorrelationEngine.h> // for PVCorrelationEngine
#include <inendi/PVScene.h>             // for PVScene
#include <inendi/PVPythonInterpreter.h>

#include <pvkernel/core/PVDataTreeObject.h> // for PVDataTreeParent
#include <pvkernel/core/PVSerializeObject.h>

#include <sigc++/sigc++.h>

#include <QColor>
#include <QList>
#include <QString>

#include <cstdint> // for int32_t
#include <memory>  // for shared_ptr
#include <string>  // for string

namespace Inendi
{
class PVSource;
} // namespace Inendi
namespace Inendi
{
class PVView;
} // namespace Inendi
namespace PVCore
{
class PVSerializeArchiveOptions;
} // namespace PVCore

#define INENDI_ROOT_ARCHIVE_EXT "pvi"
#define INENDI_ROOT_ARCHIVE_FILTER "INENDI investigation files (*." INENDI_ROOT_ARCHIVE_EXT ")"

namespace Inendi
{
/**
 * \class PVRoot
 */
class PVRoot : public PVCore::PVDataTreeParent<PVScene, PVRoot>
{
  public:
	PVRoot();
	~PVRoot();

  public:
	bool is_empty() const { return size() == 0; }
	void clear();
	void reset_colors();

  public:
	int32_t get_new_view_id();
	QColor get_new_view_color();

  public:
	void select_view(PVView& view);
	void select_scene(PVScene& scene);
	void select_source(PVSource& source);

	Inendi::PVView* current_view() { return _current_view; }
	Inendi::PVView const* current_view() const { return _current_view; }

	Inendi::PVScene* current_scene() { return _current_scene; }
	Inendi::PVScene const* current_scene() const { return _current_scene; }

	Inendi::PVSource* current_source() { return _current_source; }
	Inendi::PVSource const* current_source() const { return _current_source; }

  public:
	void save_to_file(PVCore::PVSerializeArchive& ar);
	void load_from_archive(PVCore::PVSerializeArchive& ar);

	void set_path(QString path) { _path = path; }
	const QString& get_path() const { return _path; }

  public:
	std::string get_serialize_description() const override { return "Investigation"; }

  public:
	PVCorrelationEngine& correlations() { return _correlations; }
	const PVCorrelationEngine& correlations() const { return _correlations; }
	Inendi::PVView* process_correlation(Inendi::PVView* view);

  public:
	void view_being_deleted(Inendi::PVView* view);
	void scene_being_deleted(Inendi::PVScene* view);
	void source_being_deleted(Inendi::PVSource* view);

  public:
	// Serialization
	void serialize_write(PVCore::PVSerializeObject& so) const;

	/**
	 * Read Childs from pvi
	 *
	 * root
	 *   |->data-colletions
	 *       |-> 0
	 *       |-> 1
	 *       |-> ...
	 */
	void serialize_read(PVCore::PVSerializeObject& so);

  public:
	sigc::signal<void> _scene_updated;

  private:
	PVScene* _current_scene;
	PVSource* _current_source;
	PVView* _current_view;

  private:
	PVCorrelationEngine _correlations;
	bool _correlation_running = false;

  private:
	QString _path; //!< Path where this root is saved as an investigation.

  private:
	// View related data
	QList<QRgb> _available_colors;
	QList<QRgb> _used_colors;
	int _new_view_id = 0;
};
} // namespace Inendi

#endif /* INENDI_PVROOT_H */
