/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <functional>

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVZoneRenderingBCI_types.h>

#include <inendi/PVSelection.h>

namespace Inendi
{
class PVSelection;
} // namespace Inendi

namespace PVParallelView
{

class PVBCIDrawingBackend;
class PVZonesProcessor;
class PVZonesManager;

class PVLinesView
{
	constexpr static size_t bbits = PARALLELVIEW_ZT_BBITS;

  private:
	/**
	 * It keeps zoom information for a given zone.
	 */
	struct ZoneWidthWithZoomLevel {
		constexpr static int default_base_width = PVParallelView::ZoneBaseWidth;

		ZoneWidthWithZoomLevel()
		{
			_base_width = default_base_width;
			_base_zoom_level = 0;
		}

		ZoneWidthWithZoomLevel(int16_t base_width, int16_t base_zoom_level)
		{
			_base_width = base_width;
			_base_zoom_level = base_zoom_level;
		}

		void decrease_zoom_level();

		int16_t get_base_zoom_level();
		int16_t get_base_width();

		uint32_t get_width() const;

		void increase_zoom_level();

		void set_base_width(int16_t base_width);
		void set_base_zoom_level(int16_t base_zoom_level);

		int16_t _base_width;
		int16_t _base_zoom_level;
	};

  public:
	/**
	 * It keep BCI and Image information for a given zone.
	 */
	struct SingleZoneImages {
		std::shared_ptr<PVBCIBackendImage> sel;
		std::shared_ptr<PVBCIBackendImage> bg;

		PVZoneRenderingBCIBase_p last_zr_sel;
		PVZoneRenderingBCIBase_p last_zr_bg;

		SingleZoneImages() : last_zr_sel(), last_zr_bg() {}

		SingleZoneImages(PVBCIDrawingBackend& backend, uint32_t zone_width)
		    : last_zr_sel(), last_zr_bg()
		{
			create_image(backend, zone_width);
		}

		void create_image(PVBCIDrawingBackend& backend, uint32_t zone_width);
		void set_width(uint32_t zone_width);

		void cancel_last_sel();
		void cancel_last_bg();

		void cancel_all_and_wait();
	};

	using list_zone_images_t = std::vector<SingleZoneImages>;

  public:
	PVLinesView(PVBCIDrawingBackend& backend,
	            PVZonesManager const& zm,
	            PVZonesProcessor& zp_sel,
	            PVZonesProcessor& zp_bg,
	            QObject* img_update_receiver = nullptr,
	            uint32_t zone_width = PVParallelView::ZoneMaxWidth);

  public:
	inline PVBCIDrawingBackend& backend() const { return _backend; }

	void cancel_and_wait_all_rendering();

	inline size_t get_first_visible_zone_index() const { return _first_zone; }
	inline size_t get_last_visible_zone_index() const
	{
		return get_zone_index_from_scene_pos(_visible_view_x + _visible_view_width);
	}
	uint32_t get_left_border_position_of_zone_in_scene(size_t zone_index) const;

	size_t get_number_of_managed_zones() const;
	size_t get_number_of_visible_zones() const
	{
		return get_last_visible_zone_index() - get_first_visible_zone_index() + 1;
	}

	int32_t get_left_border_of_scene() const
	{
		return get_left_border_position_of_zone_in_scene(0);
	}
	int32_t get_right_border_of_scene() const
	{
		const size_t last_z = get_number_of_managed_zones() - 1;
		return get_left_border_position_of_zone_in_scene(last_z) + 2 * PVParallelView::AxisWidth +
		       get_zone_width(last_z);
	}

	inline SingleZoneImages& get_single_zone_images(const size_t zone_offset)
	{
		return _list_of_single_zone_images[zone_offset];
	}

	size_t get_zone_index_from_scene_pos(int32_t abs_pos) const;
	size_t get_zone_index_offset(size_t zone_index) const
	{
		assert(is_zone_drawn(zone_index));
		return zone_index - get_first_visible_zone_index();
	}

	inline const PVZonesManager& get_zones_manager() const { return _zm; }

	uint32_t get_zone_width(size_t zone_index) const;

	void decrease_base_zoom_level_of_zone(size_t zone_index);
	void decrease_global_zoom_level();

	void increase_base_zoom_level_of_zone(size_t zone_index);
	void increase_global_zoom_level();

	/**
	 * Initialize the zones width for make them get in the viewport. If there not enough
	 * space, the default zone width is used to make zones understandable.
	 *
	 * @param view_width the width of the view in which the zones should fit in
	 *
	 * @return true if the zones fit in width; otherwise false
	 */
	bool initialize_zones_width(int view_width);

	bool is_zone_drawn(size_t zone_index) const
	{
		return (zone_index >= get_first_visible_zone_index() &&
		        zone_index <= get_last_visible_zone_index());
	}

	void render_all_zones_images(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_zones_bg_image(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_zones_sel_image(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_single_zone_images(size_t zone_index, const float zoom_y);
	void render_single_zone_bg_image(size_t zone_index, const float zoom_y);
	void render_single_zone_sel_image(size_t zone_index, const float zoom_y);

	void set_zone_max_width(uint32_t w);
	bool set_zone_width(size_t zone_index, uint32_t width);
	// bool set_zone_width_and_render(PVZoneID zone_id, uint32_t width);

	void translate(int32_t view_x, uint32_t view_width, const float zoom_y);

	int update_number_of_zones(int view_x, uint32_t view_width);

	/**
	 * Get the average zones width
	 *
	 * @return the average zones width
	 */
	int get_average_zones_width() const;

	/**
	 * Reset the zones width to the given value
	 *
	 * As the width are discrete, the nearest valid value is used.=
	 *
	 * @param wanted_zone_width the wanted value
	 */
	void reset_zones_width(int wanted_zone_width);

  public:
	template <class F>
	inline bool set_all_zones_width(F const& f)
	{
		bool has_changed = false;
		for (size_t zone_index = 0; zone_index < _zones_width.size(); ++zone_index) {
			has_changed |= set_zone_width(zone_index, f(get_zone_width(zone_index)));
		}
		return has_changed;
	}

  private:
	size_t get_image_index_of_zone(size_t zone_id) const;

	void visit_all_zones_to_render(uint32_t view_width, std::function<void(size_t)> const& fzone);

	size_t set_new_view(int32_t new_view_x, uint32_t view_width);
	void set_nb_drawable_zones(size_t nb_zones);

	void do_translate(size_t previous_first_zone,
	                  size_t previous_visible_zones_count,
	                  uint32_t view_width,
	                  list_zone_images_t zone_images_copy,
	                  std::function<void(size_t)> fzone_draw);

	void left_rotate_single_zone_images(size_t s);
	void right_rotate_single_zone_images(size_t s);

	void connect_zr(PVZoneRenderingBCIBase* zr, const char* slot);
	void call_refresh_slots(size_t zone_index);

  private:
	PVBCIDrawingBackend& _backend;

	size_t _first_zone;

	QObject* _img_update_receiver;

	list_zone_images_t _list_of_single_zone_images;
	std::vector<ZoneWidthWithZoomLevel> _list_of_zone_width_with_zoom_level;

	PVZonesProcessor& _processor_sel;
	PVZonesProcessor& _processor_bg;

	int32_t _visible_view_x;
	uint32_t _visible_view_width;

	PVZonesManager const& _zm;
	uint32_t _zone_max_width;

	std::vector<uint32_t> _zones_width;
};
} // namespace PVParallelView

#endif
