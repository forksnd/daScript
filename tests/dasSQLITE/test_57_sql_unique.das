options gen2

require dastest/testing_boost public

require daslib/sql
require sqlite/sqlite_boost

[sql_table(name = "Users")]
struct User {
    @sql_primary_key Id : int
    @sql_unique Email : string
    Name : string
}

def private fixture(db : SqlRunner) {
    db |> create_table(type<User>)
    db |> insert(User(Id = 1, Email = "a@x.com", Name = "alice"))
}

[test]
def test_unique_emitted_as_portable_index(t : T?) {
    let table_ddl = _sql_create_table_sql(type<User>)
    t |> success(table_ddl |> find("\"Email\" TEXT NOT NULL UNIQUE") < 0,
        "@sql_unique is emitted through the provider-neutral index helper")
    t |> equal(_sql_create_indexes_sql(type<User>),
        "CREATE UNIQUE INDEX \"uq_Users_Email\" ON \"Users\" (\"Email\")")
    t |> equal(_::_sql_create_indexes_sql(default<User>, "UsersNext"),
        "CREATE UNIQUE INDEX \"uq_UsersNext_Email\" ON \"UsersNext\" (\"Email\")",
        "named-table helper rewrites both the table and generated index name")
}

[test]
def test_unique_index_rejects_duplicate(t : T?) {
    with_sqlite(":memory:") $(db) {
        fixture(db)
        let r = db |> try_insert(User(Id = 2, Email = "a@x.com", Name = "anne"))
        t |> success(r |> is_err, "duplicate Email should violate UNIQUE")
    }
}

[test]
def test_unique_does_not_block_distinct_emails(t : T?) {
    with_sqlite(":memory:") $(db) {
        fixture(db)
        db |> insert(User(Id = 2, Email = "b@x.com", Name = "bob"))
        let total = db |> query_scalar("SELECT COUNT(*) FROM Users", type<int>)
        t |> equal(total, 2, "different Email values insert fine")
    }
}
