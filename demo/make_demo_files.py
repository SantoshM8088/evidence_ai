"""Generate the FICTIONAL demo documents in this folder.

Run:  python demo/make_demo_files.py

Uses a tiny standard-library PDF writer (text-only, Helvetica) so no extra dependency is needed.
"""
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
BANNER = "FICTIONAL DEMO DOCUMENT - created for a hackathon prototype. Not real university rules."


def simple_pdf_bytes(pages):
    """Build a minimal text-based PDF. `pages` is a list of pages, each a list of text lines."""
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = {1: "<< /Type /Catalog /Pages 2 0 R >>",
               3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    page_ids, n = [], 4
    for lines in pages:
        body = " ".join(f"({esc(line)}) Tj T*" for line in lines)
        content = f"BT /F1 11 Tf 15 TL 50 770 Td {body} ET"
        objects[n] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                      f"/Resources << /Font << /F1 3 0 R >> >> /Contents {n + 1} 0 R >>")
        objects[n + 1] = f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}\nendstream"
        page_ids.append(n)
        n += 2
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {len(page_ids)} >>"

    out, offsets = b"%PDF-1.4\n", {}
    for i in range(1, n):
        offsets[i] = len(out)
        out += f"{i} 0 obj\n{objects[i]}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {n}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offsets[i]:010d} 00000 n \n".encode() for i in range(1, n))
    out += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def wrap(paragraphs, width=88):
    lines = []
    for para in paragraphs:
        lines.extend(textwrap.wrap(para, width) or [""])
        lines.append("")
    return lines


ELIGIBILITY_PAGES = [
    wrap([
        BANNER,
        "Northbridge Institute of Technology (fictional) - Merit Scholarship 2026-27",
        "Eligibility Rules",
        "1. Purpose. The Merit Scholarship recognises strong academic performance by students in "
        "their final year of study.",
        "2. Who may apply. Final-year B.Tech Computer Science and Engineering (CSE) students may "
        "apply for the Merit Scholarship. Students in other years or other programmes are not "
        "covered by these rules.",
    ]),
    wrap([
        BANNER,
        "3. Academic requirement. Applicants must have a minimum cumulative CGPA of 7.5 on a "
        "10-point scale, as shown on their most recent official transcript.",
        "4. One application. Each student may submit only one application for the Merit "
        "Scholarship in the 2026-27 cycle.",
        "5. How to apply. Application steps and required documents are described in the separate "
        "Merit Scholarship Application Guide.",
    ]),
]

APPLICATION_GUIDE = "\n\n".join([
    BANNER,
    "Northbridge Institute of Technology (fictional)\nMerit Scholarship 2026-27 - Application Guide",
    "Section 1. Submitting your application\n"
    "Complete the online application form on the student portal and upload the required documents.",
    "Section 2. Required documents\n"
    "Applicants must submit the following documents with the application:\n"
    "(a) an official academic transcript, and\n"
    "(b) a copy of their valid student ID card.\n"
    "Applications without both documents will not be reviewed.",
    "Section 3. Queries\n"
    "All scholarship queries are handled by the Merit Scholarship Review Committee. "
    "Please do not contact individual faculty members about scholarship decisions.",
    "Section 4. Deadlines\n"
    "The application deadline is published in the separate Merit Scholarship Deadline Notice.",
]) + "\n"


def deadline_pages(deadline):
    return [wrap([
        BANNER,
        "Northbridge Institute of Technology (fictional) - Merit Scholarship Deadline Notice",
        f"Application deadline: The last date to submit Merit Scholarship applications for the "
        f"2026-27 cycle is {deadline}. Late applications will not be accepted.",
        "Committee contact: Merit Scholarship Review Committee - Coordinator: Dr. Asha Verma, "
        "Email: msrc@northbridge.example, Office: Room B-214, Administration Block. "
        "Office hours: Monday to Friday, 10:00 to 16:00.",
    ])]


def main():
    (HERE / "1_eligibility_rules.pdf").write_bytes(simple_pdf_bytes(ELIGIBILITY_PAGES))
    (HERE / "2_application_guide.txt").write_text(APPLICATION_GUIDE, encoding="utf-8")
    (HERE / "3_deadline_notice_v1.pdf").write_bytes(simple_pdf_bytes(deadline_pages("10 December 2026")))
    (HERE / "3_deadline_notice_v2.pdf").write_bytes(simple_pdf_bytes(deadline_pages("15 December 2026")))
    print("Demo files written to", HERE)


if __name__ == "__main__":
    main()
