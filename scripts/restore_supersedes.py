"""
Restore script for supersession relationships and decision statuses.
Reads from data/supersedes_backup.json and restores the exact state in Neo4j.
DO NOT RUN unless roll-back/restoration of the legacy 240 implicit edges is explicitly needed.
"""

import json
import sys
from pathlib import Path
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from settings import get_settings


def restore():
    settings = get_settings()
    backup_path = PROJECT_ROOT / "data" / "supersedes_backup.json"
    if not backup_path.exists():
        print(f"Error: Backup file {backup_path} not found.")
        sys.exit(1)

    with open(backup_path, "r", encoding="utf-8") as f:
        backup = json.load(f)

    edges = backup.get("edges", [])
    decisions = backup.get("decisions", [])
    print(f"Loaded {len(edges)} edges and {len(decisions)} decision statuses from backup.")

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password)
    )

    with driver.session() as session:
        def _tx_restore(tx):
            # 1. Delete all current SUPERSEDES edges
            del_res = tx.run("MATCH ()-[r:SUPERSEDES]->() DELETE r RETURN count(r) AS deleted")
            deleted_count = del_res.single()["deleted"]
            print(f"Deleted {deleted_count} existing SUPERSEDES edges.")

            # 2. Recreate edges from backup
            create_q = """
            UNWIND $edges AS edge
            MATCH (new_d:Decision {id: edge.new_id}), (old_d:Decision {id: edge.old_id})
            MERGE (new_d)-[r:SUPERSEDES]->(old_d)
            SET r.reason = edge.reason
            RETURN count(r) AS created
            """
            create_res = tx.run(create_q, {"edges": edges})
            created_count = create_res.single()["created"]
            print(f"Created {created_count} SUPERSEDES edges.")

            # 3. Restore statuses from backup
            status_q = """
            UNWIND $decisions AS dec
            MATCH (d:Decision {id: dec.id})
            SET d.status = dec.status
            RETURN count(d) AS updated
            """
            status_res = tx.run(status_q, {"decisions": decisions})
            updated_count = status_res.single()["updated"]
            print(f"Restored status for {updated_count} decisions.")

        session.execute_write(_tx_restore)

    driver.close()
    print("Restore complete.")


if __name__ == "__main__":
    restore()
