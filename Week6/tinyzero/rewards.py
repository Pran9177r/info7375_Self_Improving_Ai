import re
from typing import Dict, Any, Optional
import math

def extract_final_answer(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from model output, prioritizing specific formats.
    Handles integers and potentially floats, removes commas.

    Args:
        text: Model's generated text

    Returns:
        Extracted answer as a float, or None if not found/invalid.
    """
    # 1. Prioritize LaTeX boxed answers: \boxed{ANSWER}
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
    if boxed_match:
        answer_str = boxed_match.group(1).replace(",", "").strip()
        try:
            return float(answer_str)
        except ValueError:
            pass # Continue searching other patterns if boxed content isn't a number

    # 2. Prioritize patterns like "The final answer is X", "= X" at the end, etc.
    #    Use lookarounds for potentially better context. Match end-of-string or common terminators.
    patterns = [
        r"(?:final answer is|the answer is|result is|it equals|it's)\s*:?\s*(-?[\d,]+(?:\.\d+)?)", # Handles floats too
        r"=\s*(-?[\d,]+(?:\.\d+)?)\s*(?:[\.\?!]|$)", # Match '=' at end or before punctuation
    ]
    # Search in reverse order of common appearance (e.g., '=' might be last)
    for pattern in reversed(patterns):
        # Find all matches, take the last one as most likely final answer
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer_str = matches[-1].replace(",", "").strip()
            try:
                return float(answer_str)
            except ValueError:
                continue # Try next pattern if this fails

    # 3. Fallback: Last number in the entire string (less reliable)
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        last_num_str = numbers[-1].replace(",", "").strip()
        try:
            return float(last_num_str)
        except ValueError:
            pass

    return None # No valid number found

def check_multiplication_cot(text: str, num1: int, num2: int, correct_answer: int) -> bool:
    """
    Basic CoT check: Does the text explicitly contain the correct calculation string?
    e.g., "8 * 3 = 24" or "8 x 3 is 24"
    """
    # Create patterns allowing different multiplication symbols and spacing
    pattern1 = rf"{num1}\s*[\*x]\s*{num2}\s*=\s*{correct_answer}"
    pattern2 = rf"{num1}\s*times\s*{num2}\s*(?:is|=)\s*{correct_answer}"
    # Add more variations as needed

    if re.search(pattern1, text) or re.search(pattern2, text, re.IGNORECASE):
        return True
    return False

def compute_reward(
    generated_text: str,
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    require_cot: bool = False # Set to True to require the CoT step
) -> float:
    """
    Compute reward for generated solution, with optional basic CoT check.

    Args:
        generated_text: Model's generated solution
        problem: Original problem dictionary (must include 'num1', 'num2' for CoT)
        tolerance: Tolerance for numerical answers (for countdown)
        require_cot: If True, reward is 1.0 only if BOTH final answer and CoT check pass.

    Returns:
        reward: 1.0 if correct (and CoT check passes if required), 0.0 otherwise.
    """
    predicted_answer_num = extract_final_answer(generated_text)

    if predicted_answer_num is None:
        return 0.0 # No valid final answer found

    final_answer_correct = False
    cot_step_correct = False # Assume false initially

    # --- Check Final Answer ---
    if problem['task'] == 'multiplication':
        correct_answer = problem['answer']
        # Use math.isclose for robust float comparison, effectively integer check here
        if math.isclose(predicted_answer_num, correct_answer, abs_tol=0.01):
             final_answer_correct = True

    elif problem['task'] == 'countdown':
        target = problem['target']
        # Use relative tolerance for countdown
        if target != 0 and math.isclose(predicted_answer_num, target, rel_tol=tolerance):
             final_answer_correct = True
        elif target == 0 and math.isclose(predicted_answer_num, target, abs_tol=tolerance): # Handle target=0 case
             final_answer_correct = True

    # --- Check CoT (if applicable) ---
    if problem['task'] == 'multiplication' and 'num1' in problem and 'num2' in problem:
        # Perform the basic CoT check
        cot_step_correct = check_multiplication_cot(
            generated_text,
            problem['num1'],
            problem['num2'],
            problem['answer']
        )

    # --- Determine Final Reward ---
    if require_cot:
        # Both final answer AND CoT step must be correct
        if final_answer_correct and cot_step_correct:
            return 1.0
        else:
            return 0.0
    else:
        # Only the final answer needs to be correct (original behavior)
        # Optional: Give partial credit or bonus if CoT is present? (more complex)
        if final_answer_correct:
            return 1.0
        else:
            return 0.0

# --- Test reward function ---
if __name__ == "__main__":
    # Test multiplication
    problem1 = {
        'task': 'multiplication',
        'num1': 8, 'num2': 3, # Add numbers for CoT check
        'answer': 24,
        'prompt': 'What is 8 × 3?'
    }

    # Correct answer + Correct CoT
    text1 = "Let me calculate: 8 * 3 = 24. The final answer is \\boxed{24}."
    reward1_no_cot = compute_reward(text1, problem1, require_cot=False)
    reward1_req_cot = compute_reward(text1, problem1, require_cot=True)
    print(f"Test 1 (Correct + CoT): reward (no CoT req) = {reward1_no_cot}, reward (CoT req) = {reward1_req_cot}") # 1.0, 1.0

    # Correct answer + Missing/Incorrect CoT
    text2 = "The result is 24."
    reward2_no_cot = compute_reward(text2, problem1, require_cot=False)
    reward2_req_cot = compute_reward(text2, problem1, require_cot=True)
    print(f"Test 2 (Correct, no CoT): reward (no CoT req) = {reward2_no_cot}, reward (CoT req) = {reward2_req_cot}") # 1.0, 0.0

    # Incorrect answer + Correct CoT (unlikely but possible)
    text3 = "Let's see: 8 * 3 = 24. Therefore, the answer is 25."
    reward3_no_cot = compute_reward(text3, problem1, require_cot=False)
    reward3_req_cot = compute_reward(text3, problem1, require_cot=True)
    print(f"Test 3 (Incorrect, CoT present): reward (no CoT req) = {reward3_no_cot}, reward (CoT req) = {reward3_req_cot}") # 0.0, 0.0

    # Incorrect answer + No CoT
    text4 = "I think it is 11."
    reward4_no_cot = compute_reward(text4, problem1, require_cot=False)
    reward4_req_cot = compute_reward(text4, problem1, require_cot=True)
    print(f"Test 4 (Incorrect, no CoT): reward (no CoT req) = {reward4_no_cot}, reward (CoT req) = {reward4_req_cot}") # 0.0, 0.0

    # Test extraction robustness
    text5 = "The calculation is 8 x 3 which equals 24." # Fallback extraction
    reward5 = compute_reward(text5, problem1)
    print(f"Test 5 (Fallback extraction): reward = {reward5}") # Should be 1.0