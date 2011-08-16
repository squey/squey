include_directories(AFTER ${PVKERNEL_INCLUDE_DIRS})
include_directories(AFTER ${PICVIZ_INCLUDE_DIRS})
include_directories(AFTER ${ICU_INCLUDE})

add_definitions(-DLIBPVGL -DDEBUG)

# Choose the wanted Window Toolkit
# available: FREEGLUT3 only for the moment
add_definitions(-DUSE_WTK_FREEGLUT3)

if(WIN32)
        set(PVGL_SHARE_DIR "share/")
else(WIN32)
        set(PVGL_SHARE_DIR "share/")
#        set(PVGL_SHARE_DIR "${CMAKE_INSTALL_PREFIX}/share/picviz/")
endif(WIN32)
add_definitions(-DPVGL_SHARE_DIR="${PVGL_SHARE_DIR}")

add_subdirectory(src)

SET(PVGL_DATA_SHADERS_PARALLEL_FILES
data/shaders/parallel/axis_bg.frag
data/shaders/parallel/axis_bg.vert
data/shaders/parallel/axis.frag
data/shaders/parallel/axis.vert
data/shaders/parallel/lines_map_mask.frag
data/shaders/parallel/lines_map_mask.vert
data/shaders/parallel/lines_unselected.frag
data/shaders/parallel/lines_unselected.geom
data/shaders/parallel/lines_unselected.vert
data/shaders/parallel/lines_zombie_fbo.frag
data/shaders/parallel/lines_zombie_fbo.vert
data/shaders/parallel/lines_zombie.frag
data/shaders/parallel/lines_zombie.geom
data/shaders/parallel/lines_zombie.vert
data/shaders/parallel/lines_fbo_axis_mode.frag
data/shaders/parallel/lines_fbo_axis_mode.vert
data/shaders/parallel/lines_map_fbo.frag
data/shaders/parallel/lines_map_fbo.vert
data/shaders/parallel/lines_fbo.frag
data/shaders/parallel/lines_fbo.vert
data/shaders/parallel/lines.frag
data/shaders/parallel/lines.geom
data/shaders/parallel/lines.vert
data/shaders/parallel/selection.frag
data/shaders/parallel/selection.vert
)

SET(PVGL_DATA_SHADERS_SCATTER_FILES
data/shaders/scatter/dot.frag
data/shaders/scatter/dot.vert
data/shaders/scatter/selection.frag
data/shaders/scatter/selection.vert
)

SET(PVGL_DATA_FILES
	data/FreeSans.ttf
	data/gui.png
)

SET(PVGL_DATA_SHADERS_PARALLEL_FILES
data/shaders/parallel/axis_bg.frag
data/shaders/parallel/axis_bg.vert
data/shaders/parallel/axis.frag
data/shaders/parallel/axis.vert
data/shaders/parallel/lines_map_mask.frag
data/shaders/parallel/lines_map_mask.vert
data/shaders/parallel/lines_unselected.frag
data/shaders/parallel/lines_unselected.geom
data/shaders/parallel/lines_unselected.vert
data/shaders/parallel/lines_zombie_fbo.frag
data/shaders/parallel/lines_zombie_fbo.vert
data/shaders/parallel/lines_zombie.frag
data/shaders/parallel/lines_zombie.geom
data/shaders/parallel/lines_zombie.vert
data/shaders/parallel/lines_fbo_axis_mode.frag
data/shaders/parallel/lines_fbo_axis_mode.vert
data/shaders/parallel/lines_map_fbo.frag
data/shaders/parallel/lines_map_fbo.vert
data/shaders/parallel/lines_fbo.frag
data/shaders/parallel/lines_fbo.vert
data/shaders/parallel/lines.frag
data/shaders/parallel/lines.geom
data/shaders/parallel/lines.vert
data/shaders/parallel/selection.frag
data/shaders/parallel/selection.vert
)

SET(PVGL_DATA_SHADERS_SCATTER_FILES
data/shaders/scatter/dot.frag
data/shaders/scatter/dot.vert
data/shaders/scatter/selection.frag
data/shaders/scatter/selection.vert
)

SET(PVGL_DATA_FILES
	data/FreeSans.ttf
	data/gui.png
)

SET(PVGL_DATA_SHADERS_PARALLEL_FILES
data/shaders/parallel/axis_bg.frag
data/shaders/parallel/axis_bg.vert
data/shaders/parallel/axis.frag
data/shaders/parallel/axis.vert
data/shaders/parallel/lines_map_mask.frag
data/shaders/parallel/lines_map_mask.vert
data/shaders/parallel/lines_unselected.frag
data/shaders/parallel/lines_unselected.geom
data/shaders/parallel/lines_unselected.vert
data/shaders/parallel/lines_zombie_fbo.frag
data/shaders/parallel/lines_zombie_fbo.vert
data/shaders/parallel/lines_zombie.frag
data/shaders/parallel/lines_zombie.geom
data/shaders/parallel/lines_zombie.vert
data/shaders/parallel/lines_fbo_axis_mode.frag
data/shaders/parallel/lines_fbo_axis_mode.vert
data/shaders/parallel/lines_map_fbo.frag
data/shaders/parallel/lines_map_fbo.vert
data/shaders/parallel/lines_fbo.frag
data/shaders/parallel/lines_fbo.vert
data/shaders/parallel/lines.frag
data/shaders/parallel/lines.geom
data/shaders/parallel/lines.vert
data/shaders/parallel/selection.frag
data/shaders/parallel/selection.vert
)

SET(PVGL_DATA_SHADERS_SCATTER_FILES
data/shaders/scatter/dot.frag
data/shaders/scatter/dot.vert
data/shaders/scatter/selection.frag
data/shaders/scatter/selection.vert
)

SET(PVGL_DATA_FILES
	data/FreeSans.ttf
	data/gui.png
)

# INSTALL(FILES ${PVGL_DATA_SHADERS_PARALLEL_FILES} DESTINATION ${PVGL_SHARE_DIR}/shaders/parallel/)
# INSTALL(FILES ${PVGL_DATA_SHADERS_SCATTER_FILES} DESTINATION ${PVGL_SHARE_DIR}/shaders/scatter/)
# INSTALL(FILES ${PVGL_DATA_FILES} DESTINATION ${PVGL_SHARE_DIR})
