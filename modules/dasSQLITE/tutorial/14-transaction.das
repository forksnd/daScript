options gen2
// based on https://zetcode.com/db/sqlitec/

require sqlite/sqlite_boost
require daslib/stringify

[export]
def main {
    var db : sqlite3?
    var rc = sqlite3_open(":memory:", unsafe(addr(db)))
    if (rc != SQLITE_OK) {
        to_log(LOG_ERROR, "Cannot open database: sqlite3_errmsg(db)\n")
        sqlite3_close(db)
        return
    }
    let sql = %stringify~
        DROP TABLE IF EXISTS Friends;
        BEGIN TRANSACTION;
        CREATE TABLE Friends(Id INTEGER PRIMARY KEY, Name TEXT);
        INSERT INTO Friends(Name) VALUES ('Tom');
        INSERT INTO Friends(Name) VALUES ('Rebecca');
        INSERT INTO Friends(Name) VALUES ('Jim');
        INSERT INTO Friends(Name) VALUES ('Robert');
        COMMIT;
    %%
    var err_msg : string
    rc = sqlite3_exec(db, sql, unsafe(addr(err_msg)))
    if (rc != SQLITE_OK) {
        to_log(LOG_ERROR, "SQL error: {err_msg}\n")
        sqlite3_free(err_msg)
        sqlite3_close(db)
        return
    }
    let last_id = sqlite3_last_insert_rowid(db)
    to_log(LOG_INFO, "The last Id of the inserted row is {last_id}\n")
    sqlite3_close(db)
}
