from decimal import Decimal
import tempfile
import unittest
from pathlib import Path

from pagila_parity_demo import CUTOFF, bucket_diff, fold_source, project_dump, reproduce


def dump(records):
    body = []
    for name, header, line in records:
        body += [f"COPY public.{name} ({header}) FROM stdin;", line, r"\."]
    return ("\n".join(body) + "\n").encode()


class PagilaDemoTests(unittest.TestCase):
    def setUp(self):
        self.fixture = [
            ("inventory", "inventory_id, film_id, store_id, last_update", "10\t3\t2\t2007-03-15"),
            ("rental", "rental_id, inventory_id, customer_id, staff_id, last_update, rental_period",
             "20\t10\t5\t4\t2007-03-15\tunused"),
            ("payment_p2007_03", "payment_id, customer_id, staff_id, rental_id, amount, payment_date",
             "30\t5\t4\t20\t0.99\t2007-03-15 01:00:00"),
            ("payment_p2007_04", "payment_id, customer_id, staff_id, rental_id, amount, payment_date",
             f"31\t5\t4\t20\t1.99\t{CUTOFF}"),
        ]

    def test_projection_includes_all_payment_partitions_and_independent_fold_excludes_cutoff(self):
        projected = project_dump(dump(self.fixture))
        self.assertEqual(projected["inventory"], [["10", "2"]])
        self.assertEqual(projected["rental"], [["20", "10"]])
        self.assertEqual(len(projected["payment"]), 2)
        self.assertEqual(fold_source(projected), {("2", "2007-03-15"): (1, Decimal("0.99"))})

    def test_missing_copy_end_fails_closed(self):
        cut = dump(self.fixture).rsplit(b"\\.", 1)[0]
        with self.assertRaisesRegex(ValueError, "incomplete fixture"):
            project_dump(cut)

    def test_changed_header_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected header"):
            project_dump(dump(self.fixture).replace(b"film_id, store_id", b"store_id, film_id"))

    def test_duplicate_source_key_fails_closed(self):
        projected = project_dump(dump(self.fixture))
        projected["inventory"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate inventory"):
            fold_source(projected)

    def test_duplicate_payment_fails_closed_even_outside_cutoff(self):
        projected = project_dump(dump(self.fixture))
        projected["payment"] += [projected["payment"][-1]]
        with self.assertRaisesRegex(ValueError, "duplicate payment"):
            fold_source(projected)

    def test_missing_boundary_suppresses_comparison_result(self):
        key = ("2", "2007-03-15")
        source, target = {key: (1, Decimal("0.99"))}, {key: (1, Decimal("1.99"))}
        self.assertEqual(bucket_diff(source, target, None)["status"], "UNRESOLVED")
        self.assertEqual(bucket_diff(source, target, {"fixture_commit": "test"})["status"], "MISMATCH")

    def test_changed_or_missing_bucket_is_reported(self):
        key = ("2", "2007-03-15")
        source = {key: (1, Decimal("0.99"))}
        self.assertEqual(bucket_diff(source, {}, {"fixture_commit": "test"})["differences"], [
            {"date": "2007-03-15", "source": [1, "0.99"], "store": 2, "target": None}
        ])

    def test_modified_fixture_is_rejected_before_binary_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pagila-schema.sql").write_text("changed", encoding="utf-8")
            (root / "pagila-data.sql").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixture checksum mismatch"):
                reproduce(root, root)
