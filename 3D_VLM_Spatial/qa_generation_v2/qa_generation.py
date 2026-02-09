import os
import csv
import json
from openai import OpenAI
from tqdm import tqdm

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your refined, concise prompt
BASE_PROMPT = """
You are a medical AI assistant specialized in radiology and 3D spatial reasoning.

Read the 3D CT scan report below and generate 7–10 question–answer (QA) pairs 
that test a vision–language model’s understanding of spatial and anatomical relationships 
explicitly described in the report.

Focus only on spatial facts such as:
- Laterality (right vs. left, unilateral vs. bilateral)
- Vertical position (upper/lower, superior/inferior)
- Anterior–posterior relations
- Medial–lateral orientation (central/peripheral/midline)
- Spatial extent or boundaries (confined to, crossing midline, extending into)
- Adjacency or containment (within, posterior to, adjacent to)

Guidelines:
- Use only information from the Findings and Impressions sections.
- Do not include diagnostic, interpretive, or normality statements.
- Questions must emphasize *where*, *which side*, *above/below*, or *extent*.
- Answers must be strictly factual and directly derived from the report.

Output as a valid JSON list:
[
  {"question": "...", "answer": "..."},
  ...
]
"""

# ----------------------------
# ✅ MAIN FUNCTION
# ----------------------------
def generate_spatial_QA(findings, impressions):
    """Generate spatial QA pairs from a CT report using OpenAI API."""
    full_prompt = (
        BASE_PROMPT
        + f"\n\nHere is the 3D CT Scan Report:\n**Findings:** {findings}\n**Impressions:** {impressions}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or "gpt-4o" for higher accuracy
        messages=[
            {"role": "system", "content": "You are a precise radiology assistant."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    try:
        qa_pairs = json.loads(content)
    except json.JSONDecodeError:
        qa_pairs = [{"error": "Invalid JSON output", "raw_output": content}]
    return qa_pairs


# ----------------------------
# ✅ PROCESS CSV
# ----------------------------
def process_csv(input_csv, output_json):
    results = []
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing CT Reports"):
            file_name = row.get("file_name", "")
            findings = row.get("findings_en", "")
            impressions = row.get("impressions_en", "")

            # Check for multi-variant pattern (e.g., volume_1_a_1 / _2)
            base_name = "_".join(file_name.split("_")[:-1])
            variant_id = file_name.split("_")[-1].replace(".nii.gz", "")

            qa_output = generate_spatial_QA(findings, impressions)

            results.append({
                "file_name": file_name,
                "base_volume": base_name,
                "variant": variant_id,
                "qa_pairs": qa_output
            })

    # Save combined results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} QA entries to {output_json}")


# ----------------------------
# ✅ ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate spatial QA pairs for CT reports using OpenAI API.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--output_json", default="spatial_QA_output.json", help="Path to save generated JSON output.")
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_json)
 