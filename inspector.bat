@echo off

:SET BUILD_MODE=Debug
SET BUILD_MODE=RelWithDebInfo
:SET BUILD_MODE=Release

:SET PICVIZ_LOG_FILE="debug.txt"
SET PICVIZ_DEBUG_LEVEL=INFO
:SET PICVIZ_DEBUG_FILE="debug.txt"
SET CACTUSLABS_TRUNK_DIR=E:\cactuslabs\trunk

SET PICVIZ_LAYER_FILTERS_DIR=libpicviz\plugins\layer-filters\%BUILD_MODE%\
SET PICVIZ_PLOTTING_FILTERS_DIR=libpicviz\plugins\plotting-filters\%BUILD_MODE%\
SET PICVIZ_MAPPING_FILTERS_DIR=libpicviz\plugins\mapping-filters\%BUILD_MODE%\
SET PICVIZ_FUNCTIONS_DIR=libpicviz\plugins\functions\%BUILD_MODE%\
SET PICVIZ_FILTERS_DIR=libpicviz\plugins\filters\%BUILD_MODE%\
SET PVGL_SHARE_DIR=libpvgl\data\
SET PVRUSH_NORMALIZE_DIR=libpvrush\plugins\normalize\%BUILD_MODE%\
SET PVRUSH_INPUTTYPE_DIR=libpvrush\plugins\input_types\%BUILD_MODE%\
SET PVRUSH_SOURCE_DIR=libpvrush\plugins\sources\%BUILD_MODE%\
SET PVRUSH_NORMALIZE_HELPERS_DIR=libpvrush\plugins\normalize-helpers\
SET PVFILTER_NORMALIZE_DIR=libpvfilter\plugins\normalize\%BUILD_MODE%\

PATH=%PATH%;%CACTUSLABS_TRUNK_DIR%\libpvcore\src\%BUILD_MODE%;%CACTUSLABS_TRUNK_DIR%\libpvrush\src\%BUILD_MODE%;%CACTUSLABS_TRUNK_DIR%\libpicviz\src\%BUILD_MODE%;%CACTUSLABS_TRUNK_DIR%\libpvgl\src\%BUILD_MODE%;%CACTUSLABS_TRUNK_DIR%\libpvfilter\src\%BUILD_MODE%;C:\dev\tbb\bin\ia32\vc9;

echo Trunk directory is: %CACTUSLABS_TRUNK_DIR%

%CACTUSLABS_TRUNK_DIR%\picviz-inspector\src\%BUILD_MODE%\picviz-inspector.exe





