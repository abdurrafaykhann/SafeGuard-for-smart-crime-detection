"""Fill the complaints database with FICTIONAL demo data for the public fraud check.

    python scripts/seed_demo_complaints.py [--replace] [--seed 3]

Every business name is invented and every web address uses a reserved test domain (.example), so nothing
here refers to a real organisation. Also imports the old domain list in safeguard/data/fraud_agencies.json
(placeholder domains) as confirmed complaints. Records created by this script have created_by='demo-seed'."""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (name, domain, type, category, how many complaints, summary fragments)
ENTITIES = [
    ("Brightpath Holdings Ltd", "brightpath-holdings.example", "business", "Investment scam", 7, ["promised 30% monthly returns", "refused withdrawals after deposit", "phone number stopped working"]),
    ("QuickProfit Invest", "quickprofit-invest.example", "website", "Investment scam", 11, ["fake trading dashboard showed profits", "asked for fees to release funds", "impersonated a licensed broker"]),
    ("Silverline Parcel Services", "silverline-parcels.example", "business", "Non-delivery", 5, ["paid for delivery that never arrived", "customer service unreachable", "tracking number was invalid"]),
    ("Kestrel Home Repairs", None, "business", "Advance-fee fraud", 4, ["took deposit then never started the work", "cold-called elderly residents", "used a false licence number"]),
    ("Harbourview Charity Appeal", "harbourview-appeal.example", "website", "Fake charity", 6, ["collected donations for a disaster that did not exist", "no registered charity number"]),
    ("Meridian Visa Consultants", "meridian-visa.example", "agency", "Immigration fraud", 9, ["charged for guaranteed visas", "forged embassy letters", "disappeared after payment"]),
    ("Lumen Tech Support", "lumen-support.example", "business", "Tech-support scam", 8, ["pop-up claimed the computer was infected", "demanded gift cards", "took remote control of a laptop"]),
    ("Oakridge Prize Draw", "oakridge-prizes.example", "website", "Lottery scam", 6, ["told winner to pay a release fee", "no draw ever held"]),
    ("Zenith Crypto Exchange", "zenith-crypto.example", "website", "Cryptocurrency scam", 13, ["wallet balance frozen until a 'tax' was paid", "fake celebrity endorsements", "withdrawals blocked"]),
    ("Pinecrest Loans", "pinecrest-loans.example", "business", "Loan scam", 5, ["requested upfront insurance fee for a loan", "no lending licence"]),
    ("Vantage Job Placement", "vantage-jobs.example", "agency", "Job scam", 4, ["charged training fees for jobs that did not exist", "collected passport copies"]),
    ("Coral Bay Rentals", "coralbay-rentals.example", "website", "Rental scam", 5, ["listing used stolen photos", "deposit taken for a property the 'owner' did not have"]),
    ("Northgate Online Pharmacy", "northgate-pharmacy.example", "website", "Counterfeit goods", 3, ["sent unlabelled pills", "no pharmacist contact"]),
    ("Greenfield Solar Co", "greenfield-solar.example", "business", "Advance-fee fraud", 3, ["deposit for panels never installed"]),
    ("Atlas Dating Connect", "atlas-dating.example", "website", "Romance scam", 4, ["profiles asked members for money", "fake identity photos"]),
    # dismissed / resolved only -> shows "caution" rather than "unsafe"
    ("Willowbrook Furniture", "willowbrook-furniture.example", "business", "Non-delivery", 2, ["late delivery; refunded after contact", "dispute resolved"]),
    ("Fairway Driving School", None, "business", "Misleading advertising", 1, ["advertised pass rate not substantiated"]),
]
DISMISSED_ONLY = {"Willowbrook Furniture", "Fairway Driving School"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    from safeguard import create_app
    from safeguard.extensions import db
    from safeguard.models import Complaint
    from safeguard.services.complaints import name_key

    app = create_app()
    rng = random.Random(a.seed)
    now = datetime(2026, 11, 1)
    with app.app_context():
        if a.replace:
            Complaint.query.filter_by(created_by="demo-seed").delete()
            db.session.commit()
        n = Complaint.query.count()

        def add(name, domain, etype, cat, summary, status, days_ago):
            nonlocal n
            n += 1
            db.session.add(Complaint(reference=f"FR-{(now - timedelta(days=days_ago)).year}-{n:04d}", entity_name=name, name_key=name_key(name),
                                     entity_domain=domain, entity_type=etype, category=cat, summary=summary[:400], status=status,
                                     filed_at=now - timedelta(days=days_ago), created_by="demo-seed"))

        for name, domain, etype, cat, count, notes in ENTITIES:
            for i in range(count):
                if name in DISMISSED_ONLY:
                    status = "dismissed"
                else:
                    status = rng.choices(["reported", "under_investigation", "confirmed"], [3, 3, 2])[0]
                add(name, domain, etype, cat, rng.choice(notes).capitalize() + ".", status, rng.randint(5, 700))
        path = Path(__file__).resolve().parent.parent / "safeguard" / "data" / "fraud_agencies.json"
        for domain, cat in json.loads(path.read_text(encoding="utf-8")).items():
            if not Complaint.query.filter_by(entity_domain=domain).first():
                add(domain, domain, "website", cat, "Listed on the legacy fraud-domain register.", "confirmed", rng.randint(60, 900))
        db.session.commit()
        print(f"Complaints in database: {Complaint.query.count()} across {db.session.query(Complaint.name_key).distinct().count()} entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
