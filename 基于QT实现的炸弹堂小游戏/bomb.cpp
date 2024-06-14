#include "bomb.h"
#include <QPixmap>
#include <QGraphicsScene>
#include "destructiblewall.h"
#include "explosion.h"
#include "player.h"
#include "QDebug"

Bomb::Bomb(QGraphicsScene* scene, QGraphicsItem* parent)
    : QObject(), QGraphicsPixmapItem(parent), explosion(nullptr)
{
    QPixmap bombImage(":/img/bomb.png");
    setPixmap(bombImage.scaled(50, 50));

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &Bomb::explode);
    timer->start(2000);
}


void Bomb::explode()
{
    Explosion* explosion = new Explosion(scene(), nullptr);
    explosion->setPos(x(), y());
    scene()->addItem(explosion);

    connect(explosion, &Explosion::expired, explosion, &Explosion::deleteLater);
    QTimer::singleShot(1000, explosion, &Explosion::handleExpire);

    QList<QGraphicsItem*> adjacentItems = scene()->items(QRectF(x() - 50, y() - 50, 150, 150));
    bool died = false;
    Player* survivingPlayer = nullptr;
    foreach (QGraphicsItem* item, adjacentItems) {
        DestructibleWall* wall = dynamic_cast<DestructibleWall*>(item);
        if (wall) {
            scene()->removeItem(wall);
        }

        Player* player = dynamic_cast<Player*>(item);
        if (player) {
            died = true;
            survivingPlayer = player;
            scene()->removeItem(player);
        }
    }

    if (died) {
        resultScene = new QGraphicsScene();
        resultScene->setBackgroundBrush(Qt::black);

        QGraphicsTextItem* message = new QGraphicsTextItem();
        message->setPlainText("The " + survivingPlayer->getPlayerName() + " survived!");
        message->setDefaultTextColor(Qt::black);
        message->setFont(QFont("Chilanka", 24));
        message->setPos(100, 100);
        resultScene->addItem(message);

        playButton = new QPushButton();
        playButton->setText("Play again?");
        playButton->setGeometry(QRect(230,420,100,47));
        connect(playButton, SIGNAL("&QPushButton::clicked"), this, SLOT("&Bomb::handlePlayButton"));

        resultScene->addWidget(playButton);

        scene()->setParent(nullptr);
        scene()->deleteLater();
        scene()->views().first()->setScene(resultScene);
    }

    scene()->removeItem(this);
    deleteLater();
}

void Bomb::handlePlayButton()
{
    qDebug() << "Exit button clicked!";
    QApplication::quit();

//    resultScene->addWidget(actionReboot);
    }

void Bomb::handleExitButtonClicked()
{
    qDebug() << "Exit button clicked!";
    QApplication::quit();
}

void Bomb::handleExplosionFinished()
{
    if (explosion)
    {
        scene()->removeItem(explosion);
        explosion->deleteLater();
        explosion = nullptr;
    }

    emit expired();
    deleteLater();
}

void Bomb::handleExplosionExpired()
{
    emit expired();
    Explosion* explosion = qobject_cast<Explosion*>(sender());
    if (explosion)
    {
        explosion->deleteLater();
    }
}
