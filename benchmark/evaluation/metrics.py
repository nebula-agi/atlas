"""
Metrics module for evaluating answer quality.

Uses LLM-based binary scoring by default:
- 1 = Correct (prediction matches gold answer semantically)
- 0 = Incorrect (prediction does not match gold answer)
"""

import re
import os
import requests
from typing import Dict, Any, Optional, List, Tuple
from difflib import SequenceMatcher

# Try importing optional LLM dependencies for semantic evaluation
try:
    from google import genai
    from google.genai import types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# LLM configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_MODEL = "gemini-3.0-pro" 
OPENROUTER_MODEL = "openai/gpt-4o"


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    - Lowercase
    - Strip whitespace
    - Remove punctuation
    - Normalize common variations
    """
    if not answer:
        return ""
    
    # Lowercase and strip
    normalized = answer.lower().strip()
    
    # Remove common punctuation
    normalized = re.sub(r'[.,!?;:\'"()\[\]{}]', '', normalized)
    
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Normalize common answer variations
    normalized = normalized.replace("i don't know", "unknown")
    normalized = normalized.replace("i do not know", "unknown")
    normalized = normalized.replace("not mentioned", "unknown")
    normalized = normalized.replace("not specified", "unknown")
    normalized = normalized.replace("cannot determine", "unknown")
    normalized = normalized.replace("no information", "unknown")
    
    return normalized.strip()


def compute_answer_score(
    prediction: str,
    gold_answer: Dict[str, Any],
    answer_type: str,
    use_llm_eval: bool = True  # Now defaults to True
) -> Dict[str, Any]:
    """
    Compute the score for a predicted answer against the gold answer.
    
    Uses LLM-based binary scoring by default:
    - 1 = Correct (LLM judge determines prediction matches gold)
    - 0 = Incorrect (LLM judge determines prediction does not match)
    
    Args:
        prediction: The predicted answer string
        gold_answer: Dict containing 'text' and optionally 'supporting_items'
        answer_type: One of 'short_answer', 'verbatim', 'boolean', 'abstain'
        use_llm_eval: Whether to use LLM for evaluation (default: True)
    
    Returns:
        Dict with 'score' (0 or 1), 'match_type', and other metadata
    """
    gold_text = gold_answer.get("text", "")
    
    # Use LLM judge for binary scoring (default)
    if use_llm_eval:
        score = llm_judge_binary(prediction, gold_text, answer_type)
        return {
            "score": score,
            "match_type": "llm_judge",
            "answer_type": answer_type,
            "gold_text": gold_text,
            "prediction": prediction,
        }
    
    # Fallback to simple heuristic scoring if LLM not available
    return compute_heuristic_score(prediction, gold_text, answer_type)


def llm_judge_binary(prediction: str, gold: str, answer_type: str) -> int:
    """
    Use an LLM judge to determine if the prediction matches the gold answer.
    
    Returns:
        1 if correct, 0 if incorrect
    """
    # Build the prompt based on answer type
    if answer_type == "abstain":
        prompt = f"""You are a judge evaluating if a model correctly abstained from answering when the information was not available.

Gold Answer: {gold}
Model's Answer: {prediction}

The model should indicate it doesn't know or cannot answer (e.g., "Unknown", "I don't know", "Not mentioned", etc.)

Does the model correctly indicate uncertainty or lack of knowledge? Answer with ONLY "YES" or "NO".

Answer:"""
    elif answer_type == "boolean":
        prompt = f"""You are a judge evaluating if a model's yes/no answer matches the gold answer.

Gold Answer: {gold}
Model's Answer: {prediction}

Does the model's answer match the gold answer (both yes/true OR both no/false)? Answer with ONLY "YES" or "NO".

Answer:"""
    elif answer_type == "verbatim":
        prompt = f"""You are a judge evaluating if a model correctly recalled a specific piece of information.

Gold Answer: {gold}
Model's Answer: {prediction}

Mark as correct if the model's answer contains the gold answer or is semantically equivalent.
Minor additions, qualifiers, or rephrasing are acceptable as long as the core information matches.

Does the model's answer contain the correct information? Answer with ONLY "YES" or "NO".

Answer:"""
    else:  # short_answer
        prompt = f"""You are a judge evaluating if a model's answer is correct.

Gold Answer: {gold}
Model's Answer: {prediction}

IMPORTANT - Be lenient. Mark as correct if:
- The model's answer contains the gold answer (e.g., "Office wifi password" contains "wifi password")
- The answers are semantically equivalent even if phrased differently
- The core information matches, even with additional context or qualifiers
- Names, terms, or key facts are the same
- Different formats conveying the same meaning (e.g., "Name + Year + !" is equivalent to "name followed by year followed by exclamation point")
- Formulas, shorthand, or symbolic notation matches prose descriptions of the same thing

Only mark as incorrect if the model gives a fundamentally different or wrong answer.

Is the model's answer correct? Answer with ONLY "YES" or "NO".

Answer:"""

    try:
        response = call_llm_judge(prompt)
        
        # Parse binary response
        response_normalized = response.strip().upper()
        
        if "YES" in response_normalized:
            return 1
        elif "NO" in response_normalized:
            return 0
        else:
            # If unclear, fall back to heuristic
            print(f"[LLM Judge] Unclear response: {response}, falling back to heuristic")
            result = compute_heuristic_score(prediction, gold, answer_type)
            return 1 if result["score"] >= 0.5 else 0
            
    except Exception as e:
        print(f"[LLM Judge] Error: {e}, falling back to heuristic")
        result = compute_heuristic_score(prediction, gold, answer_type)
        return 1 if result["score"] >= 0.5 else 0


def get_google_client():
    """
    Create a Google GenAI client using API key or Application Default Credentials.
    Prioritizes API key if available, falls back to Vertex AI ADC.
    """
    if not GOOGLE_AVAILABLE:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    else:
        # Use Application Default Credentials for Vertex AI
        return genai.Client(vertexai=True)


def call_llm_judge(prompt: str) -> str:
    """
    Call the LLM to get a judgment.
    Priority: Google (Gemini 2.5 Flash via API key or Vertex) > OpenRouter > OpenAI
    """
    # Priority 1: Google (API key or Vertex ADC)
    if GOOGLE_AVAILABLE:
        try:
            client = get_google_client()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                    temperature=0.1,
                ),
            )
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                return part.text.strip()
            return ""
        except Exception as e:
            # Fall through to other providers if Google fails
            print(f"[LLM Judge] Google error: {e}, trying fallback...")
    
    # Priority 2: OpenRouter with GPT-4o (fallback)
    if os.environ.get("OPENROUTER_API_KEY"):
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.1,
        }
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()
    
    # Priority 3: OpenAI direct
    elif OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    
    raise ValueError("No LLM credentials available for judging. Set GOOGLE_API_KEY (or Vertex ADC), OPENROUTER_API_KEY, or OPENAI_API_KEY")


def compute_heuristic_score(prediction: str, gold: str, answer_type: str) -> Dict[str, Any]:
    """
    Fallback heuristic scoring when LLM is not available.
    Returns a dict with score and match_type.
    """
    pred_normalized = normalize_answer(prediction)
    gold_normalized = normalize_answer(gold)
    
    if answer_type == "abstain":
        # Check if prediction indicates uncertainty
        abstain_terms = {
            "unknown", "i don't know", "i do not know", "not sure",
            "cannot determine", "no information", "not mentioned",
            "not specified", "unclear", "uncertain", "n/a", "none",
            "not available", "cannot answer", "don't have information"
        }
        for term in abstain_terms:
            if term in pred_normalized:
                return {"score": 1.0, "match_type": "correctly_abstained"}
        return {"score": 0.0, "match_type": "incorrectly_answered"}
    
    elif answer_type == "boolean":
        positive_terms = {"yes", "true", "correct", "right", "affirmative", "1"}
        negative_terms = {"no", "false", "incorrect", "wrong", "negative", "0"}
        
        gold_is_positive = any(term in gold_normalized for term in positive_terms)
        gold_is_negative = any(term in gold_normalized for term in negative_terms)
        pred_is_positive = any(term in pred_normalized for term in positive_terms)
        pred_is_negative = any(term in pred_normalized for term in negative_terms)
        
        if (gold_is_positive and pred_is_positive) or (gold_is_negative and pred_is_negative):
            return {"score": 1.0, "match_type": "boolean_match"}
        return {"score": 0.0, "match_type": "boolean_mismatch"}
    
    else:  # short_answer or verbatim
        # Exact match
        if pred_normalized == gold_normalized:
            return {"score": 1.0, "match_type": "exact"}
        
        # Contains gold
        if gold_normalized in pred_normalized:
            return {"score": 1.0, "match_type": "contains_gold"}
        
        # String similarity
        similarity = SequenceMatcher(None, pred_normalized, gold_normalized).ratio()
        if similarity >= 0.8:
            return {"score": 1.0, "match_type": "high_similarity"}
        
        return {"score": 0.0, "match_type": "no_match"}


def aggregate_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate scores across multiple evaluation results.
    
    Returns summary statistics including:
    - Overall accuracy
    - Accuracy by answer type
    - Accuracy by pillar
    """
    if not results:
        return {"overall_accuracy": 0.0, "count": 0}
    
    total_score = sum(r.get("score", 0) for r in results)
    count = len(results)
    
    # By answer type
    by_answer_type = {}
    for r in results:
        atype = r.get("answer_type", "unknown")
        if atype not in by_answer_type:
            by_answer_type[atype] = {"total": 0, "count": 0}
        by_answer_type[atype]["total"] += r.get("score", 0)
        by_answer_type[atype]["count"] += 1
    
    for atype in by_answer_type:
        cnt = by_answer_type[atype]["count"]
        by_answer_type[atype]["accuracy"] = by_answer_type[atype]["total"] / cnt if cnt > 0 else 0
    
    # By pillar (if available)
    by_pillar = {}
    for r in results:
        pillar = r.get("pillar", "unknown")
        if pillar not in by_pillar:
            by_pillar[pillar] = {"total": 0, "count": 0}
        by_pillar[pillar]["total"] += r.get("score", 0)
        by_pillar[pillar]["count"] += 1
    
    for pillar in by_pillar:
        cnt = by_pillar[pillar]["count"]
        by_pillar[pillar]["accuracy"] = by_pillar[pillar]["total"] / cnt if cnt > 0 else 0
    
    return {
        "overall_accuracy": total_score / count if count > 0 else 0.0,
        "total_score": total_score,
        "count": count,
        "by_answer_type": by_answer_type,
        "by_pillar": by_pillar,
    }
