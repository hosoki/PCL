#include "pclviewer.h"
#include <QApplication>
#include <QMainWindow>
#include <QSplashScreen>
#include <QBitmap>

int main (int argc, char *argv[])
{
  QApplication a (argc, argv);

  QPixmap pxmSplash( "../data/splashimg.png" );
  QSplashScreen wndSplash( pxmSplash );
  wndSplash.setMask( pxmSplash.mask() );
  wndSplash.show();
  a.processEvents();

  PCLViewer w;
  w.show ();

  return a.exec ();
}
