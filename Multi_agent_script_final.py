

# BLOCK 1: IMPORTS AND CONFIGURATION
import pandas as pd
import openai
import numpy as np
import os
import json
import re
import time # Added to handle delays between retries

import tiktoken

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


openai.api_key = "myOpenAI_key"

# Dynamic paths (will be requested from the user at runtime)
DEALS_PATH = "/Users/lorenzoloria/Desktop/multi_agent/annunci.xlsx"
KNOWLEDGE_PATH = "/Users/lorenzoloria/Desktop/multi_agent/corrected_merged_direction_pvalue_counts.xlsx"
LOG_DIR = "/Users/lorenzoloria/Desktop/multi_agent/logs"
OUTPUT_PATH = "/Users/lorenzoloria/Desktop/multi_agent/predictions.xlsx"

# GPT model to use (can be changed to control costs)
GPT_MODEL = "gpt-4o"

np.random.seed(42)
os.makedirs(LOG_DIR, exist_ok=True)

# END BLOCK

# BLOCK 2: LOAD INPUT FILES
deals_df = pd.read_excel(DEALS_PATH)
knowledge_df = pd.read_excel(KNOWLEDGE_PATH)

# Normalize column names
knowledge_df.columns = [col.strip().lower() for col in knowledge_df.columns]

# Check that the 'variable' column exists
if "variable" not in knowledge_df.columns:
    raise ValueError("La colonna 'variable' è mancante nel file della knowledge base.")

# Convert variables to lowercase (prevents matching errors)
knowledge_df["variable"] = knowledge_df["variable"].str.lower()

# Create a new numeric score column from the 'p_value' column
def extract_score(pval_str):
    if isinstance(pval_str, str):
        if "1%" in pval_str:
            return 1.0
        elif "5%" in pval_str:
            return 0.8
        elif "10%" in pval_str:
            return 0.5
    return 0.0  # default: not significant or missing

knowledge_df["significance_score"] = knowledge_df["p_value"].apply(extract_score)

# Filter only empirically active signals (source_column > 0)
knowledge_df = knowledge_df[knowledge_df["source_column"] > 0].copy()

# Check that the essential columns are present in the deals file
for col in ["id", "text"]:
    if col not in deals_df.columns:
        raise ValueError(f"Missing column: {col} in deals_df")
    
# END BLOCK

# BLOCK 2B: LOAD HISTORICAL CAR DATA FOR AGGREGATE STATS ONLY
CAR_STATS_PATH = "/Users/lorenzoloria/Desktop/multi_agent/CAR_ground storico.xlsx"

car_data = pd.read_excel(CAR_STATS_PATH)[["CAR Totale", "CAR_Target", "CAR_Acquirer"]]

car_summary = {
    "mean_total": car_data["CAR Totale"].mean(),
    "perc_total_positive": (car_data["CAR Totale"] > 0).mean() * 100,
    "mean_target": car_data["CAR_Target"].mean(),
    "mean_acquiror": car_data["CAR_Acquirer"].mean(),
    "std_total": car_data["CAR Totale"].std(),
    "median_total": car_data["CAR Totale"].median(),
    "q25_total": car_data["CAR Totale"].quantile(0.25),
    "q75_total": car_data["CAR Totale"].quantile(0.75),
}

historical_note = (
    f"Based on 105 historical M&A deals:\n"
    f"- Mean total CAR: {car_summary['mean_total']:.2f}%\n"
    f"- Median total CAR: {car_summary['median_total']:.2f}%\n"
    f"- Standard deviation: {car_summary['std_total']:.2f}%\n"
    f"- Interquartile range: {car_summary['q25_total']:.2f}% to {car_summary['q75_total']:.2f}%\n"
    f"- % of positive total CARs: {car_summary['perc_total_positive']:.1f}%\n"
    f"- Average CAR for targets: {car_summary['mean_target']:.2f}%\n"
    f"- Average CAR for acquirors: {car_summary['mean_acquiror']:.2f}%\n\n"
)
# END BLOCK

# BLOCK 2C: LOAD SEMANTIC DICTIONARY
with open("/Users/lorenzoloria/Desktop/multi_agent/dizionario_semantico_finale_mna.json", "r") as f:
    semantic_dict = json.load(f)

def extract_semantic_signals(text):
    """
    Corrected version that only searches for whole words to avoid false positives.
    Uses regular expressions with word boundaries (\b).
    """
    text_lower = text.lower()
    found = []
    already_found_words = set()

    for polarity in ["positive", "negative"]:
        for category in ["keywords", "verbs", "modals", "themes"]:
            for word in semantic_dict.get(polarity, {}).get(category, []):
                
                if word in already_found_words:
                    continue

                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                
                if re.search(pattern, text_lower):
                    score = {"keywords": 0.5, "verbs": 0.8, "modals": 0.8, "themes": 1.0}.get(category, 0.5)
                    found.append((word, polarity, score))
                    already_found_words.add(word)
    return found

def format_semantic_signals(signals):
    if not signals:
        return "No semantic risk signals detected."
    sorted_signals = sorted(signals, key=lambda x: abs(x[2]), reverse=True)[:10]
    return "Semantic signals from historical data:\n" + "\n".join(
        [f"- '{w}' → {lbl} (score: {sc:.2f})" for w, lbl, sc in sorted_signals]
    )

# END BLOCK


# BLOCK 2D: INTEGRAZIONE KNOWLEDGE BASE NEL DATAFRAME

# Filter relevant rows (source_column > 0)
knowledge_df = knowledge_df[knowledge_df["source_column"] > 0].copy()

# Normalize variable names and directions
knowledge_df["variable"] = knowledge_df["variable"].str.lower()
knowledge_df["direction"] = knowledge_df["direction"].str.lower()

# Create a column like: 'premium_price_positive'
knowledge_df["kb_column"] = knowledge_df["variable"] + "_" + knowledge_df["direction"]

# Combine empirical strength and statistical significance
knowledge_df["combined_score"] = knowledge_df["source_column"] * knowledge_df["significance_score"]

# Initialize columns in the deals dataframe
for colname in knowledge_df["kb_column"].unique():
    if colname not in deals_df.columns:
        deals_df[colname] = 0.0


# === VARIABLE SCALE REFERENCE ===
# These specifications clarify the expected format and scale of each variable used in the empirical logic.
# The following structured variables are used in empirical association checks.
# All values in the dataset are expected to follow these formats:
# - premium_price: expressed as decimal (e.g., 0.25 = 25%)
# - relative_target_size: decimal (target mktcap / acquiror mktcap)
# - leverage_acquiror: decimal (total debt / total assets)
# - gov_index_acquiror: integer (e.g., 0–10 scale)
# - cash_payment_dummy: binary (1 if cash-financed, else 0)
# - sic_dummy: binary (1 = same industry, else 0)
# - nation_dummy: binary (1 = same country, else 0)
# - mktcap_acquiror: in thousands of EUR (e.g., 1600 = €1.6 billion)


# Define thresholds for activation
thresholds = {
    "premium_price": 0.30,
    "gov_index_acquiror": 7,
    "cash_payment_dummy": 0.5,
    "relative_target_size": 0.30,
    "mktcap_acquiror": 1600,
    "leverage_acquiror": 0.30,
    "sic_dummy": 0.5,
    "nation_dummy": 0.5
       
}

# Apply activation logic based on thresholds
for _, row in knowledge_df.iterrows():
    base_var = row["variable"]
    direction = row["direction"]
    score = row["combined_score"]
    colname = row["kb_column"]

    if base_var in deals_df.columns:
        threshold = thresholds.get(base_var, 0)  # Default threshold = 0 if not specified

        if direction == "positive":
            deals_df.loc[deals_df[base_var] > threshold, colname] = score
        elif direction == "negative":
            deals_df.loc[deals_df[base_var] > threshold, colname] = score

# END BLOCK

# BLOCK 3: THEORY PROMPT
presupposed_theory = (
    "You are participating in a debate to estimate the probability that the total CAR "
    "(a weighted average of the acquirer's and target's CARs) is positive in a [-3;+3] day window "
    "around the deal announcement date. Use the deal text, contextual knowledge, and reasoning.\n\n"

    "The goal is to predict CAR_positive — a binary variable equal to 1 if the total CAR is positive, and 0 otherwise.\n\n"

    "The total CAR is defined as:\n"
    "CAR_Totale = (CAR_Target × MktCap_Target + CAR_Acquiror × MktCap_Acquiror) / (MktCap_Target + MktCap_Acquiror)\n\n"

    "Market capitalizations are measured 6 trading days before the announcement date.\n"
    "MktCap_Target is adjusted to exclude any pre-announcement ownership (toehold) already held by the acquiror.\n\n"

    "Most past deals involve an acquiror that is larger than the target in terms of market capitalization. "
    "As a result, the acquiror’s CAR tends to dominate the direction of the total CAR.\n\n"

    "**All deals in this dataset are from European markets.** When estimating the probability of value creation, "
    "consider characteristics specific to the European M&A context.\n\n"

    "Your estimated probability should reflect the interplay between these structural priors, deal-specific facts, and empirical evidence."
)

themes_guidance = (
    "=== THEMATIC GUIDANCE ===\n"
    "In the Macro-Themes round, your task is to evaluate the announcement based on its factual and strategic content, "
    "as extracted directly from the text. Focus on the clarity, coherence, and relevance of the strategic rationale presented.\n\n"

    "Key areas to consider include: the strategic fit between acquiror and target, the presence of articulated synergies "
    "(especially cost synergies), the payment structure (cash vs equity), the regulatory landscape, and the integration plan.\n\n"

    "Signals associated with positive market reactions typically include:\n"
    "- Clear articulation of synergies with quantifiable or operational detail\n"
    "- Strategic logic tied to specific goals (e.g., market expansion, technology acquisition)\n"
    "- Mention of integration experience or a clear post-deal implementation roadmap\n"
    "- Confidence in deal execution reflected in statements about financing or timing\n"
    "- Regulatory approval already obtained or deal already approved by shareholders\n\n"

    "Conversely, the following red flags are frequently associated with negative CARs:\n"
    "- Vague or generic justifications such as 'growth' without a clear mechanism\n"
    "- Absence of any mention of synergies or integration considerations\n"
    "- No indication of cultural, operational, or organizational alignment\n"
    "- Overemphasis on deal size or ambition without concrete strategic framing\n"
    "- Deal explicitly subject to multiple complex conditions, antitrust scrutiny, or uncertain regulatory approval\n"
    "- Language that reveals uncertainty about timing, closing, or stakeholder support\n\n"

    "You must ignore tone, sentiment, or speculative language in this round. Focus exclusively on the **substantive content** "
    "of the extracted facts. Treat omissions of key elements (e.g., no mention of synergies or regulatory path) as meaningful.\n\n"

    "Additionally, evaluate the **cultural fit** between the acquiror and the target based on their respective countries. "
    "Draw upon your knowledge of European business cultures. Cross-border deals between culturally distant countries "
    "(e.g., Southern vs. Northern Europe, Anglo-Saxon vs. Continental Europe) often face integration risks and investor skepticism, "
    "especially when cultural aspects are not addressed. Conversely, same-country deals or transactions between culturally aligned regions "
    "(e.g., Germany–Austria, Nordics) tend to inspire greater confidence due to smoother integration expectations.\n\n"

    "Your evaluation should reflect whether the announcement provides a well-grounded, credible strategic justification for the transaction, "
    "and whether there are clear signals of execution readiness. Conversely, you should penalize announcements that appear underdeveloped, "
    "highly contingent, or strategically ambiguous."
    
)

semantic_guidance = (
    "=== SEMANTIC GUIDANCE ===\n"
    "In the Semantic Focus round, your task is to analyze the announcement text based exclusively on linguistic and tonal cues. "
    "You must ignore structured variables and empirical literature. Base your judgment solely on the language used in the announcement.\n\n"

    "The tone, choice of words, and degree of assertiveness carry important signals about the firm's internal confidence and the likely market interpretation.\n\n"

    "Based on text mappings of historical M&A announcements with known CAR outcomes, the following patterns have emerged:\n\n"

    "Strongly assertive and forward-looking language — often associated with positive CARs:\n"
    "- Verbs such as: 'enhance', 'enhanced', 'drive', 'generate', 'will', 'is going to', 'accelerate', 'realize'\n"
    "- These suggest high confidence, clarity of intent, and commitment to execution.\n\n"

    "Cautiously optimistic or soft-hedged expressions — often perceived as non-committal and associated with neutral or negative CARs:\n"
    "- Verbs such as: 'foster', 'leverage', 'support', 'enable'\n"
    "- These imply potential rather than action, and are often used when strategic clarity or conviction is lower.\n\n"

    "Hedging or uncertain language — frequently associated with negative CARs:\n"
    "- Phrases like: 'seeks to', 'is expected to', 'wishes to', 'aims to', 'could', 'may', 'would', 'might'\n"
    "- These indicate tentative commitment, strategic ambiguity, or risk aversion, and often raise doubts among investors.\n\n"

    "Your role is to detect these patterns and assess the underlying confidence and credibility of the announcement. "
    "Do not simply count positive-sounding words — evaluate their assertiveness, concreteness, and tone. "
    "Statements that are overly vague or filled with conditional language should be treated with caution.\n\n"

    "Absence of key strategic terms — especially the word 'synergies' — is a strong negative signal:\n"
    "- Historical analysis of deal announcements reveals that the omission of any reference to 'synergies' is frequently associated with negative CARs.\n"
    "- This absence suggests either a lack of value creation rationale or insufficient preparation in communicating the strategic benefits of the deal.\n\n"
    "If the word 'synergies' is not mentioned at all in the announcement, you must treat this as a serious red flag.\n"
    "Such omissions significantly reduce the credibility of the deal's value proposition in the eyes of investors."

)

    
empirical_guidance = (
    "=== EMPIRICAL GUIDANCE ===\n"
    "\nPlease pay particular attention to the following variables, which have shown empirical associations "
    "with the likelihood of a positive CAR in past studies. Note that the strength and consistency of evidence varies across variables:\n\n"

    "- `cash_payment_dummy`: cash-financed deals are strongly and consistently associated with higher CARs, as they reduce adverse selection and signal confidence.\n"
    "- `gov_index_acquiror`: higher governance scores are robustly linked to stronger managerial discipline and better deal outcomes.\n"
    "- `premium_price`: high premiums are generally associated with lower CARs due to overpayment risk, unless clearly justified by synergies.\n"
    "- `relative_target_size`: larger targets relative to the acquiror tend to increase execution complexity and are often associated with lower CARs.\n"
    "- `mktcap_acquiror`: large acquirors can reflect empire-building motives leading to negative effects, but can also manage integration better.\n"
    "- `leverage_acquiror`: high financial leverage in the acquiror may indicate discipline or distress, depending on context. Empirical findings on its impact on CAR are mixed, with some studies associate high leverage with lower CAR due to financial constraints, while others find no significant effect or suggest it disciplines management in value-creating deals.\n"
    "- `sic_dummy`: deals between firms in the same industry can benefit from strategic fit and operational synergies, increasing the chance of positive CAR.\n"
    "- `nation_dummy`: same-country deals typically face fewer cultural, legal, and integration barriers, and are often associated with higher CARs.\n"
    
    
)

reasoning_guidance = (
    "=== REASONING GUIDANCE ===\n"
    "Note that some empirical signals may be ambiguous or even conflicting. "
    "For instance, while high leverage is often associated with lower CARs, some studies suggest it can reflect managerial discipline. "
    "Similarly, acquiror size may either signal capacity to integrate or empire-building motives.\n\n"
    "As the Manager, you are expected to go beyond isolated and linear interpretations of individual variables. "
    "Instead, you must apply advanced reasoning and draw upon your background knowledge of M&A dynamics.\n\n"
    "Consider how variables interact in this specific deal. For example:\n"
    "- A high premium might be acceptable if paired with strong governance and an intra-sector match.\n"
    "- Elevated leverage could be mitigated by a large acquiror and a same-country context.\n"
    "- The risks of a large relative target might be offset by operational synergies or integration experience.\n\n"
    "You are not a statistical model. You are a strategic evaluator tasked with understanding nuance, context, and trade-offs. "
    "Assess the configuration of variables as a whole—not just individually—when estimating the probability of value creation.\n\n"
    "Base your judgment on reasoning, context, and knowledge of real-world M&A dynamics.\n\n"
    "You should also assess the cultural fit between the two companies, using the countries of origin as a proxy. "
    "European M&A research suggests that greater cultural distance increases integration risks and reduces the likelihood of synergy realization. "
    "Conversely, favorable cultural alignment can enhance post-deal performance and partially offset structural weaknesses in the transaction.\n"

)

# END BLOCK

# BLOCK 4A: GPT CALLS AND KNOWLEDGE INTEGRATION (CON GESTIONE ERRORI)

def gpt_call(prompt, model=GPT_MODEL, temperature=0.5, max_tokens=500):
    """ Performs an OpenAI API call with error handling and retry attempts. """
    retries = 3
    delay = 5  # seconds
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.RateLimitError as e:
            print(f"Rate limit error. Waiting {delay} seconds before retrying... ({i+1}/{retries})")
            time.sleep(delay)
            delay *= 2 # Double the wait time (exponential backoff)
        except Exception as e:
            print(f"An API error occurred: {e}. Retrying... ({i+1}/{retries})")
            time.sleep(delay)
            delay *= 2
    print(f"Unable to get a response from the API after {retries} attempts.")
    return "ERROR: API call failed" # Fallback value in case of failure

def normalize(s):
    return re.sub(r"[\s_]+", "", s.strip().lower())

def extract_empirical_signals(deal_vars):
    """
    Extracts the empirical signals triggered by structured variables, assigning either positive or negative scores.
    """
    signal_template = "- Variable: {var} → historically associated with {direction} CAR (value: {val:.2f})"
    signals = []

    # Full list of monitored variables
    var_list = [
        "cash_payment_dummy", "gov_index_acquiror", "premium_price",
        "relative_target_size", "mktcap_acquiror", "leverage_acquiror",
        "sic_dummy", "nation_dummy"
    ]

    for var in var_list:
        for direction in ["positive", "negative"]:
            key = f"{var}_{direction}"
            val = deal_vars.get(key, 0)
            if val > 0:
                signals.append(signal_template.format(var=var, direction=direction, val=val))

    return "Empirical variables activated for this deal:\n" + "\n".join(signals) if signals else "No empirical signals found."

# END BLOCK

# BLOCK 4B: AGENT PROMPTING, ARGUMENTS, EXPERT EVALUATION

def build_agent_prompt(role, deal_variables, round_specific_input, semantic_summary, round_type="General", extracted_facts=None, themes_guidance= "", semantic_guidance= "", empirical_guidance= ""):
    header = (
        presupposed_theory +
        "\n\nYou are participating in a debate about an M&A deal involving two European firms. "
        "The deal is part of a European sample, so your assessment should consider the European M&A context "
        "as it affects expectations, risks, and norms around value creation.\n"
    )

    # Selective content based on round type
    
    if round_type == "Macro-Themes":
        info_block = (
            f"\n=== EXTRACTED FACTS ===\n{round_specific_input}\n\n"
            f"=== INTERPRETATION GUIDANCE ===\n{themes_guidance}\n"
        )

    elif round_type == "Semantic Focus":
        info_block = (
            f"\n=== SEMANTIC SIGNALS ===\n{semantic_summary}\n\n"
            f"=== INTERPRETATION GUIDANCE ===\n{semantic_guidance}\n"
        )

    elif round_type == "Empirical Associations":
        info_block = (
            f"\n=== STRUCTURED VARIABLES ===\n{json.dumps(deal_variables)}\n"
            f"=== EMPIRICAL SIGNALS ===\n{round_specific_input}\n\n"
            f"=== INTERPRETATION GUIDANCE ===\n{empirical_guidance}\n"
        )
        
    else:
        # Fallback with everything included
        extracted_theme_block = (
            "\n=== THEMES (Macro) ===\n" + "\n".join(extracted_facts) + "\n"
        ) if extracted_facts else ""
        info_block = (
            extracted_theme_block +
            f"\n=== SEMANTIC SIGNALS ===\n{semantic_summary}\n"
            f"\n=== STRUCTURED VARIABLES ===\n{json.dumps(deal_variables)}\n"
            f"\n=== EMPIRICAL SIGNALS ===\n{round_specific_input}\n\n"
            f"\n=== THEMATIC GUIDANCE ===\n{themes_guidance}\n"
            f"\n=== SEMANTIC GUIDANCE ===\n{semantic_guidance}\n"
            f"\n=== EMPIRICAL GUIDANCE ===\n{empirical_guidance}\n"
        )

    # Instruction specific to the round type
    round_instruction = {
        "Macro-Themes": (
            "Focus exclusively on the key factual elements of the deal, such as strategic rationales, synergies, payment type, premium, regulatory aspects, etc., "
            "as extracted from the announcement text. Do not perform sentiment analysis. Do not rely on empirical associations or prior literature."
        ),
        "Semantic Focus": (
            "Focus exclusively on the language and tone of the announcement. Rely on sentiment, word choice, modality, and strategic framing. "
            "Do not refer to structured variables (e.g., premium_price, payment method) or to extracted factual content. "
            "Do not include empirical evidence from past studies."
        ),
        "Empirical Associations": (
            "Focus exclusively on structured variables in the deal (e.g., cash_payment_dummy, gov_index_acquiror, premium_price, relative_target_size, mktcap_acquiror, leverage_acquiror, sic_dummy, nation_dummy), and their historical associations with CAR as documented in the literature. "
            "Use only data from structured fields and do not mention sentiment, tone, or narrative elements from the announcement text."
        )
    }.get(round_type, "Use all available information.")

    # Role-specific instruction
    task_instruction = {
        "Proponent": "As the Proponent, argue why this deal is likely to result in a positive total CAR.",
        "Opponent": "As the Opponent, argue why this deal may not generate value. Focus on risks and weaknesses.",
        "Expert": "As the Expert, assess each argument as either Supported or Speculative."
    }[role]

    return header + "\n\n" + info_block + "\n\n" + round_instruction + "\n\n" + task_instruction


def agent_argument(text, role, deal_vars, round_specific_input, semantic_summary,extracted_facts=None, missing_elements=None, round_type="General", themes_guidance=None, semantic_guidance=None, empirical_guidance=None):
    risk_summary = ""
    if missing_elements and missing_elements != ["None missing"]:
        risk_summary = "\n\n⚠️ Warning: The following key elements are missing or vague in the announcement:\n- " + "\n- ".join(missing_elements)

    prompt = (
        build_agent_prompt(role, deal_vars, round_specific_input, semantic_summary, round_type=round_type, extracted_facts=extracted_facts, themes_guidance=themes_guidance, semantic_guidance=semantic_guidance, empirical_guidance=empirical_guidance) +
        risk_summary +
        "\n\nAnnouncement text:\n" + text
    )
    return gpt_call(prompt, temperature=0.7, max_tokens=700)

def expert_evaluation(argument):
    prompt = (
    f"Evaluate the following argument from the debate:\n\n{argument}\n\n"
    "Step 1: Classify the argument as either 'Supported' or 'Speculative'.\n"
    "→ 'Supported' means the reasoning is based on **explicit, verifiable facts** from the deal text.\n"
    "→ 'Speculative' means it relies on **general claims, assumptions, or vague language** without firm grounding in the announcement.\n\n"
    "Step 2: Justify your classification in one short paragraph.\n"
    "Respond in this exact format:\n"
    "Label: <Supported/Speculative>\nJustification: <your explanation>"
)
    response = gpt_call(prompt, temperature=0.3)
    label_match = re.search(r"Label:\s*(Supported|Speculative)", response, re.IGNORECASE)
    just_match = re.search(r"Justification:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
    return {
        "label": label_match.group(1) if label_match else "Speculative",
        "justification": just_match.group(1).strip() if just_match else "No justification provided"
    }

def extract_facts_llm(text):
    prompt = (
        "From the following M&A announcement, extract a list of 3 to 7 distinct, non-overlapping key facts or themes. "
        "Each fact should capture a meaningful aspect of the deal (e.g., synergies, premium, payment type, etc.).\n\n"
        f"{text}\n\n-"
    )

    response = gpt_call(prompt, temperature=0.3, max_tokens=500)
    facts = [line.strip("- ").strip() for line in response.strip().split("\n") if line.strip()]
    return facts

def validate_facts(facts):
    filtered = [f for f in facts if len(f.strip()) > 10 and "deal" not in f.lower()]
    return filtered if filtered else ["No valid facts extracted."]

# END BLOCK

# BLOCK 5A: STRUCTURED VARIABLE FLAGS WITH THRESHOLDS

def check_structural_aspects(deal_vars):
    checks = {
        "cash payment": lambda dv: dv.get("cash_payment_dummy", 0) == 1,
        "high governance index": lambda dv: dv.get("gov_index_acquiror", 0) > 7,
        "high premium": lambda dv: dv.get("premium_price", 0) > 30,
        "large target": lambda dv: dv.get("relative_target_size", 0) > 30,
        "large acquiror": lambda dv: dv.get("mktcap_acquiror", 0) > 1600,
        "high leverage": lambda dv: dv.get("leverage_acquiror", 0) > 30,
        "same sector (sic_dummy)": lambda dv: dv.get("sic_dummy", 0) == 1,
        "same country (nation_dummy)": lambda dv: dv.get("nation_dummy", 0) == 1
    }
    return {aspect: fn(deal_vars) for aspect, fn in checks.items()}

# END BLOCK

# BLOCK 5B: GROUPING EXTRACTED THEMES INTO MACRO-TOPICS

def group_themes(extracted_facts):
    """
    Assigns extracted facts to one macro-theme based on keyword matching.
    Each fact is assigned to the first matching category only (no duplicates across groups).
    """

    themes = {
        "Strategic Rationale and Synergies": ["synergies", "strategy", "expansion", "growth", "alignment", "position", "leader"],
        "Financial Terms and Payment Method": ["cash", "shares", "payment", "value"],
        "Premium and Valuation": ["premium", "valuation", "price"],
        "Integration and Risks": ["integration", "culture", "execution", "challenge"],
        "Regulatory Aspects": ["regulatory", "antitrust", "approval"],
        "Finance and Leverage": ["leverage", "debt", "financing"],
        "Ownership Structure Post-Deal" : ["ownership","structure", "shareholders", "hold", "capital"],
        "Timeline and Conditions": ["apporoval", "conditions", "restrictions", ]
    }

    assigned_facts = set()
    grouped = {}

    for theme, keywords in themes.items():
        grouped[theme] = []
        for fact in extracted_facts:
            if fact in assigned_facts:
                continue  # skip if already assigned to another theme
            if any(keyword in fact.lower() for keyword in keywords):
                grouped[theme].append(fact)
                assigned_facts.add(fact)

    # Return only non-empty theme groups
    return {k: v for k, v in grouped.items() if v}

# BLOCK 6: DEBATE LOGIC AND MANAGER DECISION

def format_structural_flags(flags_struct):
    positives = [k for k, v in flags_struct.items() if v]
    return "Structured flags triggered: " + ", ".join(positives) if positives else "No structured flags detected."

def manager_decision(proponent_argument, opponent_argument, expert_eval_pro, expert_eval_opp, deal_vars, theme, presupposed_theory):
    all_arguments = f"""--- Proponent Argument ---
{proponent_argument.strip()}

→ Expert Evaluation: {expert_eval_pro.get("label", "N/A")}
→ Expert Justification: {expert_eval_pro.get("justification", "No explanation provided")}

--- Opponent Argument ---
{opponent_argument.strip()}

→ Expert Evaluation: {expert_eval_opp.get("label", "N/A")}
→ Expert Justification: {expert_eval_opp.get("justification", "No explanation provided")}
"""

    prompt = (
        presupposed_theory +
        historical_note +
        "===============================\n\n" +
        themes_guidance +
        "\n\n===============================\n\n" +
        semantic_guidance +
        "\n\n===============================\n\n" +
        empirical_guidance +
        "\n\n===============================\n\n" +
        reasoning_guidance +
        "\n\n===============================\n\n" +
        
        "You are the Manager overseeing this debate. Your goal is to provide a reasoned, non-neutral estimate "
        "of the probability (0–100%) that the total CAR will be positive, based on all available evidence.\n\n"
        "You must consider the extracted deal facts, the arguments provided by both the Proponent and the Opponent, "
        "the Expert's evaluations (Supported vs Speculative), and the empirical evidence from prior literature.\n\n"
        "You are strictly forbidden from providing vague or neutral estimates between 55% and 65%, "
        "unless the evidence is perfectly and explicitly balanced on both sides. "
        "A response of 65% is only acceptable in the rare case of total argumentative parity with no dominant signals.\n\n"
        "You are required to express a clear and decisive probability outside this neutral range "
        "whenever one side presents stronger empirical, textual, or structural support. "
        "It is your top priority to make a judgment call, not to hedge or default to ambiguity.\n\n"
        "This instruction is absolute and non-negotiable. Failure to comply with this directive "
        "will result in your output being discarded as invalid and treated as a reasoning failure.\n\n"
        f"Debate Theme: {theme}\n\n"
        f"{all_arguments}\n"
        "Respond in this exact JSON format:\n{\n  \"reasoning\": \"<your reasoning>\",\n  \"probability\": <number from 0 to 100>\n}"
    )

    response = gpt_call(prompt, temperature=0.3, max_tokens=1000)

    reasoning = "Parsing error: fallback used."
    number = 50.0

    try:
        parsed = json.loads(response)
        reasoning = parsed.get("reasoning", "").strip()
        number = float(parsed.get("probability", 50))
    except Exception:
        # Fallback using regex if JSON is invalid
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)
        probability_match = re.search(r'"probability"\s*:\s*([0-9]{1,3}(?:\.\d+)?)', response)

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        if probability_match:
            number = float(probability_match.group(1))

    # Clamp probability to the range [0, 100]
    if not (0 <= number <= 100):
        number = 50.0

    return number, reasoning
  
def run_debate_for_deal(deal_id, text, deal_vars, extracted_facts, deal, themes_guidance, semantic_guidance, empirical_guidance):
    history = []
    prev_prob = 50.0
    convergence_threshold = 5.0

    flags_struct = check_structural_aspects(deal_vars)
    flag_summary = format_structural_flags(flags_struct)
    semantic_signals = extract_semantic_signals(text)
    semantic_summary = format_semantic_signals(semantic_signals)
    missing_elements = check_missing_elements(text)

    grouped_macro = group_themes(extracted_facts)
    macro_snippet = "\n".join([f"{k}: {', '.join(v)}" for k, v in grouped_macro.items()])

    grouped_themes = {
         "Macro-Themes": macro_snippet,
         "Semantic Focus": ["Focus on the language and semantic signals in the announcement."],
         "Empirical Associations": ["Focus on the variables in this deal and their historical correlation with CAR."]
    }

    print(f"\nTemi raggruppati per il dibattito:")
    for key in grouped_themes:
        print(f"- {key}: {grouped_themes[key]}")

    for i, (macro_theme, facts) in enumerate(grouped_themes.items()):
        print(f"\n🔄 Round {i+1}/{len(grouped_themes)}: Debating topic category → {macro_theme}")

        theme = "; ".join(facts) if isinstance(facts, list) else facts

        semantic_summary = format_semantic_signals(extract_semantic_signals(text))
        empirical_snippet = extract_empirical_signals(deal_vars)
        structural_flags = format_structural_flags(check_structural_aspects(deal_vars))

        if macro_theme == "Macro-Themes":
            round_specific_input = theme
        elif macro_theme == "Semantic Focus":
            round_specific_input = semantic_summary
        elif macro_theme == "Empirical Associations":
            round_specific_input = empirical_snippet + "\n" + structural_flags
        else:
            round_specific_input = semantic_summary + "\n" + empirical_snippet + "\n" + structural_flags

        full_context_text = deal.get("announcement_text", text)
        

        proponent_arg = agent_argument(
            text=f"Full Announcement:\n{full_context_text}",
            role="Proponent",
            deal_vars=deal_vars,
            round_specific_input=round_specific_input,
            semantic_summary=semantic_summary,
            extracted_facts=extracted_facts,
            round_type=macro_theme,
            themes_guidance=themes_guidance,
            semantic_guidance=semantic_guidance,
            empirical_guidance=empirical_guidance
        )

        opponent_arg = agent_argument(
            text=f"Full Announcement:\n{full_context_text}",
            role="Opponent",
            deal_vars=deal_vars,
            round_specific_input=round_specific_input,
            semantic_summary=semantic_summary,
            extracted_facts=extracted_facts,
            round_type=macro_theme,
            themes_guidance=themes_guidance,
            semantic_guidance=semantic_guidance,
            empirical_guidance=empirical_guidance
        )

        
        expert_eval_pro = expert_evaluation(proponent_arg)
        expert_eval_opp = expert_evaluation(opponent_arg)

        prob, reasoning = manager_decision(
            proponent_arg, opponent_arg,
            expert_eval_pro, expert_eval_opp,
            deal_vars, macro_theme,
            presupposed_theory
        )

        if expert_eval_pro["label"] == "Supported" and expert_eval_opp["label"] == "Speculative":
            prob += 10
        elif expert_eval_pro["label"] == "Speculative" and expert_eval_opp["label"] == "Supported":
            prob -= 10

        # Clamp between 0 and 100 to prevent overflow
        prob = max(0, min(100, prob))


        print(f"Probabilità stimata dopo il round {i+1}: {prob:.1f}%")

        history.append({
            "round": i + 1,
            "theme": macro_theme,
            "probability": prob,
            "reasoning": reasoning,
            "proponent_arg": proponent_arg,
            "opponent_arg": opponent_arg,
            "expert_pro_eval": json.dumps(expert_eval_pro),
            "expert_opp_eval": json.dumps(expert_eval_opp),
            "missing_elements": missing_elements
        })


        prev_prob = prob if prob is not None else prev_prob
        
    # Calculate the weighted average of rounds (giving more weight to the empirical round)

    weights = {
        "Macro-Themes": 1,
        "Semantic Focus": 1,
        "Empirical Associations": 2
    }

    total_weight = sum(weights.get(entry["theme"], 1) for entry in history)
    weighted_sum = sum(entry["probability"] * weights.get(entry["theme"], 1) for entry in history)
    final_prob = weighted_sum / total_weight if total_weight > 0 else history[-1]["probability"]

    final_reasoning = (
    f"As the Manager, I have reviewed all {len(history)} rounds of this structured debate.\n"
    f"I assigned equal weight to the Macro-Themes and Semantic rounds, and double weight to the Empirical round, "
    f"given its stronger statistical grounding.\n"
    f"Based on the aggregated arguments, expert evaluations, and the distribution of signals across the rounds, "
    f"I estimate the overall probability of a positive total CAR at **{final_prob:.2f}%**.\n"
    "This figure reflects my integrated judgment across strategic, linguistic, and empirical dimensions."
)

    return final_prob, final_reasoning, history

def check_missing_elements(text):
    prompt = (
        "From the following M&A announcement, identify which of the following elements are missing or only vaguely mentioned:\n"
        "- premium"
        "- synergies, cost synergies"
        "- aligns with"
        "- integration, execution, plan\n"
        "- cash, shares, mix)\n"
        "- regulatory, risk, approval, antitrust issues\n"
        "Announcement:\n\n"
        f"{text}\n\n"
        "List only the missing or insufficiently addressed elements as bullet points. If all elements are present and clear, write: 'None missing.'"
    )
    response = gpt_call(prompt, temperature=0.3)
    if "none missing" in response.lower():
        return ["None missing"]
    missing = [line.strip("- ").strip() for line in response.strip().split("\n") if line.strip()]
    return missing

# END BLOCK

# BLOCK 8: PROCESSING LOOP AND OUTPUT (REWRITTEN AND CORRECTED)

results = []

for idx, row in deals_df.iterrows():
    deal = row
    deal_id = row["id"]
    text = row["text"]
    deal_vars = row.drop(["id", "text"]).to_dict()

    print(f"\n\n{'='*60}\nProcessing deal ID {deal_id}\n{'='*60}")

    # 1. Extract facts and analyze missing elements
    print("Step 1: Extracting key facts from announcement...")
    facts_raw = extract_facts_llm(text)
    extracted_facts = validate_facts(facts_raw)
    print(f"→ Fatti estratti: {', '.join(extracted_facts)}")
      
    print("\nStep 2: Checking for missing information...")
    missing_elements = check_missing_elements(text)
    print(f"→ Elementi mancanti o vaghi: {', '.join(missing_elements)}")

    # 2. Execute the round-based debate
    print("\nStep 3: Starting structured debate...")
    final_prob, final_reasoning, debate_history = run_debate_for_deal(
        deal_id, text, deal_vars, extracted_facts, deal, themes_guidance, semantic_guidance, empirical_guidance
    )

    # 3. Save the results
    results.append({
        "id": deal_id,
        "final_probability": round(final_prob, 2) if final_prob is not None else None,
        "manager_reasoning": final_reasoning
    })
    
    # 4. Save the detailed log for this operation
    log_path = os.path.join(LOG_DIR, f"{deal_id}_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"DEAL ANALYSIS LOG - ID: {deal_id}\n")
        f.write("="*40 + "\n\n")
        
        f.write("=== EXTRACTED FACTS ===\n")
        f.write(f"{json.dumps(extracted_facts, indent=2)}\n\n")
        f.write(f"Missing Elements Analysis: {json.dumps(missing_elements, indent=2)}\n\n")
        f.write("Grouped Macro-Themes: Used in Round 1\n\n")
        f.write("--- DEBATE HISTORY ---\n")
        for round_data in debate_history:
             f.write(f"\n--- Round {round_data['round']}: Theme: {round_data['theme']} ---\n")
             f.write(f"Round Topic Category: {round_data['theme']}\n")  # <– Qui scrive il tipo di round
             f.write(f"Manager's Probability Estimate: {round_data['probability']:.2f}%\n")
             f.write(f"Manager's Reasoning: {round_data['reasoning']}\n\n")
             f.write("Proponent Argument:\n")
             f.write(f"{round_data['proponent_arg']}\n")
             f.write(f"Expert Evaluation (Proponent): {round_data['expert_pro_eval']}\n\n")
             f.write("Opponent Argument:\n")
             f.write(f"{round_data['opponent_arg']}\n")
             f.write(f"Expert Evaluation (Opponent): {round_data['expert_opp_eval']}\n")
             f.write("-" * 20)

        f.write("\n\n--- FINAL RESULT ---\n")
        f.write(f"Final Estimated Probability: {final_prob:.2f}%\n")
        f.write(f"Final Manager Reasoning: {final_reasoning}\n")
    
    print(f"\n Deal {deal_id} processed. Detailed log saved to {log_path}")

# Save final aggregated results
final_df = pd.DataFrame(results)
final_df.to_excel(OUTPUT_PATH, index=False)
print(f"\n{'='*60}\nCompleted. Aggregated output saved to {OUTPUT_PATH}\n{'='*60}")
# END BLOCK
