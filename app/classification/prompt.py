def build_stance_prompt(claim: str, abstract: str) -> str:
    """
    Build a strict, reasoning-driven prompt for stance classification.
    """

    return f"""
You are a scientific reasoning system.

Your task is to determine the stance of a research paper abstract with respect to a given claim.

--- DEFINITIONS ---

"supports":
The paper provides evidence or findings that support the claim.

"contradicts":
The paper reports negative results, no effect, or outcomes that oppose the claim.

"neutral":
The paper is unrelated, inconclusive, or does not provide clear evidence for or against the claim.

--- RULES ---

- You MUST output ONLY valid JSON.
- Do NOT include any text outside JSON.
- Do NOT hallucinate or infer beyond the abstract.
- If the abstract does NOT directly address the claim, return "neutral".
- Do NOT treat mention of the topic as support.
- Do NOT assume causation or extrapolate beyond stated findings.
- If the abstract is unclear or insufficient → return neutral with confidence 0.0.
- Keep reasoning to maximum 2 sentences.
- Focus on the OVERALL conclusion of the study, not isolated results.

--- REASONING STEPS (MUST FOLLOW) ---

1. Identify what the CLAIM asserts (e.g., improvement, decline, or no effect).
2. Identify the OVERALL conclusion of the paper (not individual sentences).
3. Compare the two:

   - If the paper shows improvement → "supports"
   - If the paper shows no effect (e.g., "no change", "does not alter") → 
        → "contradicts" when the claim asserts improvement
   - If the paper shows decline or negative impact → "contradicts"
   - If the paper shows mixed results:
        → choose "contradicts" if any negative effects are present
        → otherwise choose "neutral"

4. Base your decision strictly on this comparison.

--- DECISION PRIORITY RULES (STRICT) ---

You MUST follow these priority rules when making the final decision:

1. If the overall conclusion indicates NO EFFECT (e.g., "no change", "does not alter", "no significant difference"):
   → ALWAYS classify as "contradicts" when the claim asserts improvement.
   → Ignore minor positive findings.

2. If BOTH positive and negative results are present:
   → If ANY negative or declining effects are reported → classify as "contradicts"

3. Do NOT override the overall conclusion with isolated positive results.

4. Only classify as "supports" if the paper clearly concludes improvement overall.

--- OUTPUT FORMAT ---

{{
  "stance": "supports" | "contradicts" | "neutral",
  "confidence": float (0 to 1),
  "reason": "short explanation"
}}

--- EXAMPLES ---

Example 1:
Claim: "Intermittent fasting improves cognitive performance."
Abstract: "Our study shows improved memory and attention in participants practicing intermittent fasting."

Output:
{{
  "stance": "supports",
  "confidence": 0.9,
  "reason": "The study reports improved cognitive performance associated with intermittent fasting."
}}

Example 2:
Claim: "Intermittent fasting improves cognitive performance."
Abstract: "No significant changes in cognitive function were observed in fasting individuals."

Output:
{{
  "stance": "contradicts",
  "confidence": 0.85,
  "reason": "The study finds no cognitive improvement from intermittent fasting."
}}

Example 3:
Claim: "Intermittent fasting improves cognitive performance."
Abstract: "Cognitive performance declined in the afternoon but remained unchanged in the morning during fasting."

Output:
{{
  "stance": "contradicts",
  "confidence": 0.85,
  "reason": "The study reports declines in cognitive performance, contradicting the claim of improvement."
}}

--- TASK ---

Claim:
{claim}

Abstract:
{abstract}
"""
