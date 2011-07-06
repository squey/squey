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

#ifndef _GKMAPVIEW_H_
#define _GKMAPVIEW_H_

#include <QWebView>

#include <geo/GKLocation.h>

typedef enum {
    GKMapTypeSatellite,
    GKMapTypeTerrain,
    GKMapTypeHybrid,
    GKMapRoadmap
} GKMapType;

class GKMapViewJsObject;
class GKMapView : public QWebView {
    Q_OBJECT

    friend class GKMapViewJsObject;

    public:
        GKMapView (QWidget *parent = 0);
        ~GKMapView();

        GKMapType mapType (void) const;
        void setMapType (GKMapType mapType);

        GKLocationCoordinate2D centerCoordinate (void) const;
        void setLocation (const QString& address);
        void setLocation (GKLocationDegrees latitude,
                          GKLocationDegrees longitude);
        void setLocation (const GKLocationCoordinate2D& coordinate);

        void addMarker (const QString& title,
                        const QString& address);
        void addMarker (const QString& title,
                        GKLocationDegrees latitude,
                        GKLocationDegrees longitude);
        void addMarker (const QString& title,
                        const GKLocationCoordinate2D& coordinate);

        void addMarkerWithWindow (const QString& title,
                                  const QString& contentString,
                                  const QString& address);
        void addMarkerWithWindow (const QString& title,
                                  const QString& contentString,
                                  GKLocationDegrees latitude,
                                  GKLocationDegrees longitude);
        void addMarkerWithWindow (const QString& title,
                                  const QString& contentString,
                                  const GKLocationCoordinate2D& coordinate);

        void removeMarker (const QString& address);
        void removeMarker (const GKLocationCoordinate2D& coordinate);
        void removeMarker (GKLocationDegrees latitude,
                           GKLocationDegrees longitude);

        void addInfoWindow (const QString& contentString,
                            const QString& address);
        void addInfoWindow (const QString& contentString,
                            GKLocationDegrees latitude,
                            GKLocationDegrees longitude);
        void addInfoWindow (const QString& contentString,
                            const GKLocationCoordinate2D& coordinate);

        void addressFromLocation (GKLocationDegrees latitude,
                                  GKLocationDegrees longitude);
        void addressFromLocation (const GKLocationCoordinate2D& coordinate);
        void locationFromAddress (const QString& address);

    public Q_SLOTS:
        void mapZoomIn (void);
        void mapZoomOut (void);
        void setMapZoom (qreal zoom);

    Q_SIGNALS:
        void locationClicked (const GKLocationCoordinate2D& coordinate);
        void locationRightClick (const GKLocationCoordinate2D& coordinate);
        void centerLocationChanged (const GKLocationCoordinate2D& coordinate);

        void markerClicked (const QString& title, const GKLocationCoordinate2D& coordinate);

        void addressFound (const QString& address, 
                           const GKLocationCoordinate2D& coordinate);
        void addressNotFound (const QString& address);

        void locationFound (const QString& address, 
                            const GKLocationCoordinate2D& coordinate);
        void locationNotFound (const GKLocationCoordinate2D& coordinate);

    private:
        GKMapViewJsObject *d;
};

#endif /* !_GKMAPVIEW_H_ */

