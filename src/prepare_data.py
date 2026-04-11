# MedQuAD Data Converter.
#
# Author
# ------
# Kishanthan Kingston
#
# Copyright
# ---------
# © 2026 Kishanthan Kingston
#
# License
# -------
# MIT
#
# Description
# -----------
# Converts MedQuAD XML files into plain text files
# readable by the RAG pipeline.

import os
import glob
from lxml import etree

INPUT_PATH = "data/MedQuAD"
OUTPUT_PATH = "data/medical_docs"


def parse_medquad_file(filepath):
    """
    Parse a MedQuAD XML file and extract question-answer pairs.

    The function reads an XML file, extracts all ``QAPair`` elements,
    and formats them into a readable text structure suitable for
    downstream RAG pipelines.

    Parameters
    ----------
    filepath : str
        Path to the MedQuAD XML file.

    Returns
    -------
    str
        A formatted string containing all extracted question-answer pairs.
        Returns an empty string if parsing fails or no valid pairs are found.

    Raises
    ------
    Exception
        Any parsing-related exception is caught internally and results
        in an empty string.
    """
    try:
        tree = etree.parse(filepath)
        root = tree.getroot()
        content = []

        for qa in root.findall(".//QAPair"):
            question = qa.findtext("Question", "").strip()
            answer = qa.findtext("Answer", "").strip()

            if question and answer:
                # Keep Q&A together in one block so they stay in the same chunk
                content.append(f"Question: {question}\nAnswer: {answer}\n---")

        return "\n\n".join(content)

    except Exception:
        return ""


def convert_all():
    """
    Convert all MedQuAD XML files into plain text files.

    This function recursively scans the input directory for XML files,
    extracts their content using ``parse_medquad_file``, and writes the
    results into the output directory as ``.txt`` files.

    Notes
    -----
    - Invalid or empty XML files are skipped.
    - Output filenames match the original XML filenames.
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    xml_files = glob.glob(f"{INPUT_PATH}/**/*.xml", recursive=True)
    print(f"Found {len(xml_files)} XML files.")

    converted = 0

    for filepath in xml_files:
        content = parse_medquad_file(filepath)
        if not content:
            continue

        filename = os.path.splitext(os.path.basename(filepath))[0]
        output_file = os.path.join(OUTPUT_PATH, f"{filename}.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        converted += 1

    print(f"Converted {converted} files to {OUTPUT_PATH}/")


if __name__ == "__main__":
    convert_all()
