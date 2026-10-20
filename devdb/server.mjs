// Local PostgreSQL + pgvector for development and tests, with no server install.
// PGlite is PostgreSQL compiled to WebAssembly; pglite-socket serves it over the
// normal wire protocol so Django and psycopg connect to it like any Postgres.
//
//   node devdb/server.mjs            # in-memory database on port 5499
//   PGLITE_DATA=./devdb/data node devdb/server.mjs   # keep data between runs
import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite-pgvector";
import { PGLiteSocketServer } from "@electric-sql/pglite-socket";

const port = Number(process.env.PGLITE_PORT ?? 5499);
const db = await PGlite.create(process.env.PGLITE_DATA, { extensions: { vector } });
await db.exec("CREATE EXTENSION IF NOT EXISTS vector");

const server = new PGLiteSocketServer({ db, port, host: "127.0.0.1", maxConnections: 8 });
await server.start();
console.log(`PGlite with pgvector listening on postgresql://postgres@127.0.0.1:${port}/postgres`);

const stop = async () => {
  await server.stop();
  await db.close();
  process.exit(0);
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
