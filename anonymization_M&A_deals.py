

!pip install --upgrade openai PyMuPDF fpdf tiktoken

from google.colab import files
import fitz, os, tempfile, zipfile, re, unicodedata
from fpdf import FPDF
from openai import OpenAI
import pandas as pd
from datetime import datetime
import tiktoken

client = OpenAI(api_key="myOpneAI_key")

def count_tokens(text, model="gpt-4o"):
    return len(tiktoken.encoding_for_model(model).encode(text))

def clean_text(text):
    text = text.replace("€", "EUR")
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

prompt_base = """
IMPORTANT: You must preserve the sentiment, rhetorical structure, and strategic tone of the announcement.

You are a text anonymization agent specialized in M&A deal announcements.

Your task is to rewrite the text below by strictly anonymizing all real-world identifiers while fully preserving the tone, sentiment, and strategic language used in the original document.

MANDATORY RULES:

1. Replace all company names with placeholders:
   - "Company A" for the acquirer
   - "Company B" for the target

2. REMOVE COMPLETELY:
   - Real company names
   - Stock exchange tickers or listings
   - Specific geographic references (countries, cities, regions)
   - Names of people, law firms, or banks
   - ISIN codes or financial symbols

3. DO NOT REMOVE:
   - Strategic or persuasive language
   - Descriptions of expected benefits, synergies, or rationale
   - Relative timing (e.g., "expected to close in six months")
   - Sentiment expressions (e.g., "transformational", "highly accretive", "strategic fit")

4. PRESERVE:
   - Financial metrics (deal value, EPS, synergies)
   - Post-deal ownership structure
   - Industry descriptions
   - Sentiment and tone

OUTPUT FORMAT:
Use clear headings:
- Summary
- Strategic Rationale
- Financial Terms
- Ownership Structure
- Timeline and Conditions

Use European business and financial style.
"""

uploaded = files.upload()
results, out_dir = [], tempfile.mkdtemp()

for file_name in uploaded:
    local_path = os.path.join(out_dir, file_name)
    with open(local_path, "wb") as f:
        f.write(uploaded[file_name])

    if file_name.endswith(".zip"):
        with zipfile.ZipFile(local_path, 'r') as zipf:
            zipf.extractall(out_dir)

pdfs = [f for f in os.listdir(out_dir) if f.endswith(".pdf")]

for pdf_name in pdfs:
    path = os.path.join(out_dir, pdf_name)
    with fitz.open(path) as doc:
        text = " ".join([p.get_text() for p in doc])

    try:
        tokens = count_tokens(prompt_base + text)

        # A. NEUTRALIZZAZIONE
        res_main = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_base},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        out_text = clean_text(res_main.choices[0].message.content)

        # B. INFERENZA SIC CODE E INDUSTRY
        res_sic = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in industry classification. Based on the M&A announcement provided, infer the most likely SIC code and industry sector for the acquiring company (Company A) and the target company (Company B). Do NOT mention real company names. Return the output as:\n\nCompany A SIC: ...\nCompany A Industry: ...\nCompany B SIC: ...\nCompany B Industry: ..."},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        sic_text = clean_text(res_sic.choices[0].message.content)

        # C. CREA PDF
        pdf_out = os.path.join(out_dir, pdf_name.replace(".pdf", "_neutralized.pdf"))
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        for line in (out_text + "\n\n[SIC Codes & Industries]\n" + sic_text).split('\n'):
            pdf.multi_cell(0, 10, line.strip())
        pdf.output(pdf_out)

        # D. LOG
        results.append({
            "File": pdf_name,
            "Tokens": tokens,
            "Out_PDF": os.path.basename(pdf_out),
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    except Exception as e:
        print(f"Errore su {pdf_name}: {e}")

# CSV con riepilogo
csv_path = os.path.join(out_dir, "results.csv")
pd.DataFrame(results).to_csv(csv_path, index=False)

# ZIP di output
final_zip = os.path.join(tempfile.gettempdir(), "output_batch_neutralized.zip")
with zipfile.ZipFile(final_zip, 'w') as zipf:
    for f in os.listdir(out_dir):
        zipf.write(os.path.join(out_dir, f), arcname=f)

# Download
files.download(final_zip)



# === EXPORT TO EXCEL ===
import pandas as pd

# Create a DataFrame with anonymized deal texts and IDs
anonymized_data = pd.DataFrame({
    'deal_id': [entry['deal_id'] for entry in results],
    'anonymized_text': [entry['anonymized'] for entry in results]
})

# Save to Excel
excel_output_path = os.path.join(tempfile.gettempdir(), "anonymized_deals.xlsx")
anonymized_data.to_excel(excel_output_path, index=False)
print(f"✅ Anonymized deals saved to: {excel_output_path}")
files.download(excel_output_path)

