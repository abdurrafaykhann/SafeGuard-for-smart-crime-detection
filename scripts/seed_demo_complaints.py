"""Fill the complaints database with FICTIONAL demo data for the public fraud check.

    python scripts/seed_demo_complaints.py [--replace] [--seed 3]

Every business name is invented and every web address uses a reserved test domain (.example), so nothing
here refers to a real organisation. See safeguard/services/demo_data.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    from safeguard import create_app
    from safeguard.extensions import db
    from safeguard.models import Complaint
    from safeguard.services.demo_data import seed_complaints

    app = create_app()
    with app.app_context():
        total = seed_complaints(a.seed, a.replace)
        print(f"Complaints in database: {total} across {db.session.query(Complaint.name_key).distinct().count()} entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
