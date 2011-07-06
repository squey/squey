/*
* Copyright (C) 2009 Matteo Bertozzi.
*
* This file is part of THLibrary.
*
* THLibrary is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* THLibrary is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with THLibrary. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _GKLOCATION_H_
#define _GKLOCATION_H_

#include <QtGlobal>

typedef qreal GKLocationDegrees;

typedef struct {
    GKLocationDegrees latitude;
    GKLocationDegrees longitude;
} GKLocationCoordinate2D;

#endif /* !_GKLOCATION_H_ */

