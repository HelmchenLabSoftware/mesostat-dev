#include "mesostatgui.h"
#include "ui_mesostatgui.h"

MesostatGui::MesostatGui(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MesostatGui)
{
    ui->setupUi(this);
}

MesostatGui::~MesostatGui()
{
    delete ui;
}
