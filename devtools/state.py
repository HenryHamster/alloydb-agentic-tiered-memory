"""Disposable schemas and portable prepared-memory snapshots; never mutate public tables."""

import re
from pathlib import Path

from psycopg2 import sql
from psycopg2.extras import Json

TABLES = (
    "documents",
    "entities",
    "memory_facts",
    "memory_entities",
    "memory_links",
    "entity_relations",
    "observations",
    "memory_conflict_reviews",
    "macro_memories",
)


class RunSchema:
    def __init__(self, connect, run_id, source):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Run schema IDs must be generated UUID hex strings")
        self.connect, self.name, self.source = connect, "dev_" + run_id, Path(source)
        self.created = False

    def connection(self):
        connection = self.connect()
        try:
            with connection.cursor() as cur:
                cur.execute(
                    sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.name))
                )
            connection.commit()
            return connection
        except BaseException:
            connection.close()
            raise

    def create(self):
        connection = self.connect()
        try:
            with connection.cursor() as cur:
                cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.name)))
                cur.execute(
                    sql.SQL("SET LOCAL search_path TO {}, public").format(sql.Identifier(self.name))
                )
                for name in ("004_retain_schema.sql", "005_reflect_schema.sql"):
                    ddl = (self.source / "sql" / name).read_text(encoding="utf8")
                    # Extensions are an admin prerequisite, not something a developer worker installs.
                    ddl = re.sub(r"CREATE EXTENSION IF NOT EXISTS [A-Za-z0-9_]+;", "", ddl)
                    cur.execute(ddl)
                # Snapshot restore includes cyclic/self-referencing fact relationships.
                cur.execute(
                    "SELECT c.relname, k.conname FROM pg_constraint k "
                    "JOIN pg_class c ON c.oid=k.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE n.nspname=%s AND k.contype='f'",
                    (self.name,),
                )
                for table, constraint in cur.fetchall():
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {}.{} ALTER CONSTRAINT {} DEFERRABLE INITIALLY DEFERRED"
                        ).format(
                            sql.Identifier(self.name),
                            sql.Identifier(table),
                            sql.Identifier(constraint),
                        )
                    )
            connection.commit()
            self.created = True
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def snapshot(self):
        result = {}
        connection = self.connection()
        try:
            with connection.cursor() as cur:
                for table in TABLES:
                    cur.execute(
                        sql.SQL("SELECT to_jsonb(t) FROM {}.{} t").format(
                            sql.Identifier(self.name), sql.Identifier(table)
                        )
                    )
                    result[table] = [row[0] for row in cur.fetchall()]
            return result
        finally:
            connection.close()

    def restore(self, snapshot):
        if set(snapshot) != set(TABLES):
            raise ValueError("Prepared snapshot has an incompatible table set")
        connection = self.connection()
        try:
            with connection.cursor() as cur:
                cur.execute("SET CONSTRAINTS ALL DEFERRED")
                for table in TABLES:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema=%s AND table_name=%s AND is_generated='NEVER' "
                        "ORDER BY ordinal_position",
                        (self.name, table),
                    )
                    columns = sql.SQL(", ").join(sql.Identifier(r[0]) for r in cur.fetchall())
                    rows = snapshot[table]
                    for offset in range(0, len(rows), 250):
                        cur.execute(
                            sql.SQL(
                                "INSERT INTO {}.{} ({}) SELECT {} FROM "
                                "jsonb_populate_recordset(NULL::{}.{}, %s::jsonb)"
                            ).format(
                                sql.Identifier(self.name),
                                sql.Identifier(table),
                                columns,
                                columns,
                                sql.Identifier(self.name),
                                sql.Identifier(table),
                            ),
                            (Json(rows[offset : offset + 250]),),
                        )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def close(self):
        if not self.created:
            return
        connection = self.connect()
        try:
            with connection.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.name)))
            connection.commit()
            self.created = False
        finally:
            connection.close()


class PrefixedValkey:
    """Keep the unchanged SessionBuffer's keys isolated between concurrent runs."""

    def __init__(self, client, run_id):
        self.client, self.prefix = client, "dev:" + run_id + ":"
        self.keys = set()

    def __getattr__(self, method):
        if method not in {"rpush", "lrange", "get", "set", "ltrim"}:
            raise AttributeError(method)

        def call(key, *args, **kwargs):
            physical = self.prefix + key
            self.keys.add(physical)
            return getattr(self.client, method)(physical, *args, **kwargs)

        return call

    def close(self):
        try:
            # Clustered Valkey rejects a multi-key DEL spanning hash slots.
            for key in self.keys:
                self.client.delete(key)
        finally:
            self.client.close()
