#!/bin/bash

# \file svn-id.sh
#
# Copyright (C) Picviz Labs 2010-2012

svn propset svn:keywords 'Id' libpvcore/src/*.cpp
svn propset svn:keywords 'Id' libpvcore/src/include/pvcore/*.h

svn propset svn:keywords 'Id' libpvfilter/src/*.cpp
svn propset svn:keywords 'Id' libpvfilter/src/include/pvfilter/*.h

svn propset svn:keywords 'Id' libpvrush/src/*.cpp
svn propset svn:keywords 'Id' libpvrush/src/include/pvrush/*.h

svn propset svn:keywords 'Id' libpicviz/src/*.cpp
svn propset svn:keywords 'Id' libpicviz/src/include/picviz/*.h

svn propset svn:keywords 'Id' libpvgl/src/*.cpp
svn propset svn:keywords 'Id' libpvgl/src/include/pvgl/*.h

svn propset svn:keywords 'Id' picviz-inspector/src/*.cpp
svn propset svn:keywords 'Id' picviz-inspector/src/include/*.h
