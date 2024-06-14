#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QPushButton>
#include <QLineEdit>
#include <QGridLayout>
#include <QGraphicsPixmapItem>
#include <QLabel>
#include "player.h"
#include "bomb.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void slotReboot();
    void placeMonster(int number, int x, int y);
    void validateNumber(const QString &text, QLineEdit *monsterInput);
    void placeMessage(const QString text, int x, int y, int size);
    void handlePlayButton();
    void handlePlayerMoved(int newX, int newY, int playerNumber);
    void handlePlaceBomb();
    void handleBombPlaced(Bomb* bomb);

signals:
    void placeBombRequested();

private:
    static int const EXIT_CODE_REBOOT = -123456789;
    Ui::MainWindow *ui;
    QGraphicsScene *scene;
    QGraphicsScene *startScene;
    QGraphicsView *view;
    QGraphicsView *startView;
    Player *player1;
    Player *player2;
    QLineEdit *chooseMonster1;
    QLineEdit *chooseMonster2;
    QPushButton *playButton;
    QGridLayout *instructionLayout;
};

#endif // MAINWINDOW_H

