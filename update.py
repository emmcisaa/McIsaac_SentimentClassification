# update.py

import sqlite3


def store_review(db_path, review, sentiment):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS reviews (review TEXT, sentiment INTEGER)')
    c.execute('INSERT INTO reviews (review, sentiment) VALUES (?, ?)', (review, sentiment))
    conn.commit()
    conn.close()
