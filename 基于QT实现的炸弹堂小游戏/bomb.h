#ifndef BOMB_H
#define BOMB_H

#include <QObject>
#include <QGraphicsPixmapItem>
#include <QTimer>
#include <QApplication>
#include <QCoreApplication>
#include <QGraphicsTextItem>
#include <QGraphicsProxyWidget>
#include <QPushButton>
#include <QGraphicsView>
#include <QProcess>




class QGraphicsScene;
class Explosion;

class Bomb : public QObject, public QGraphicsPixmapItem
{
    Q_OBJECT
public:
    Bomb(QGraphicsScene* scene, QGraphicsItem* parent = nullptr);
    void handleExplosionExpired();

private slots:
    void explode();
    void handleExplosionFinished();
    void handlePlayButton();

signals:
    void expired();

private:
    QTimer* timer;
    QAction *actionReboot;
    QGraphicsScene* resultScene;
    Explosion* explosion;
    QPushButton* playButton;
    void handleExitButtonClicked();

};

#endif // BOMB_H



