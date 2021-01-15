#include "mesostatgui.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MesostatGui w;
    w.show();

    return a.exec();
}
