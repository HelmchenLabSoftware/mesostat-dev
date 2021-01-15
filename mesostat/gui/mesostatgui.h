#ifndef MESOSTATGUI_H
#define MESOSTATGUI_H

#include <QMainWindow>

namespace Ui {
class MesostatGui;
}

class MesostatGui : public QMainWindow
{
    Q_OBJECT

public:
    explicit MesostatGui(QWidget *parent = 0);
    ~MesostatGui();

private:
    Ui::MesostatGui *ui;
};

#endif // MESOSTATGUI_H
