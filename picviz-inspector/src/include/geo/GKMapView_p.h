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

#ifndef _GKMAPVIEW_PRIVATE_H_
#define _GKMAPVIEW_PRIVATE_H_

#include <QObject>
#include <QQueue>
class QWebFrame;
class QWebPage;
class QWebView;

#include "GKLocation.h"
#include "GKMapView.h"

class GKMapViewJsObject : public QObject {
    Q_OBJECT

    public:
        GKMapViewJsObject (QObject *parent = 0);
        ~GKMapViewJsObject();

        void addJsRequest (const QString& scriptSource);
        QVariant evaluateJavaScript (const QString& scriptSource);
        void addToJavaScriptWindowObject (const QString& name, QObject *object);

        QWebView *webView (void) const;
        QWebPage *webPage (void) const;
        GKMapView *mapView (void) const;
        QWebFrame *webFrame (void) const;

    public Q_SLOTS:
        void jsLocationClicked (qreal latitude, qreal longitude);
        void jsLocationRightClicked (qreal latitude, qreal longitude);
        void jsCenterLocationChanged (qreal latitude, qreal longitude);
        void jsMarkerClicked (const QString& title, qreal latitude, qreal longitude);

        void jsAddressLocation (const QString& address, qreal latitude, qreal longitude);
        void jsAddressNotFound (const QString& address);

        void jsLocationAddress (const QString& address, qreal latitude, qreal longitude);
        void jsLocationNotFound (qreal latitude, qreal longitude);

    public:
        GKLocationCoordinate2D centerCoordinates;
        GKMapType mapType;

    private Q_SLOTS:
        void loadMapsPage (void);
        void loadFinished (bool ok);

    private:
        QQueue<QString> m_jsRequests;
        bool m_isLoaded;
};

#endif /* !_GKMAPVIEW_PRIVATE_H_ */

