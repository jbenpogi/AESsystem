#!/usr/bin/env python
"""
Test the new component-based scoring to verify:
1. ML score (0–100) is correctly scaled to content (0–50)
2. Grammar, formatting remain independent
3. Final score = content + grammar + formatting (0–100)
4. Score distribution improves (weak/average/strong essays spread apart)
"""

from scorer.ml.scoring import total_score

# Test essays
weak_essay = """
This is a weak essay. It is short and does not have much content. 
The writing is not very good. There are some grammar mistakes.
"""

average_essay = """
This essay discusses the topic reasonably well. It has some good points and 
some explanations. For example, the main idea is explained. However, there 
could be more detail. The essay has decent grammar and is formatted properly 
with multiple paragraphs.

The formatting is okay and the word count is acceptable. The essay touches 
on the main idea but could expand more. There are a few minor errors.
"""

strong_essay = """
This essay provides a comprehensive analysis of the topic. The thesis statement 
is clear and well-articulated. For example, the essay demonstrates strong 
reasoning through multiple supporting points. 

Each paragraph contains a topic sentence followed by relevant evidence and 
explanations. The author uses proper citations and builds logical connections 
between ideas. The writing is sophisticated with varied sentence structure 
and appropriate vocabulary.

The formatting is professional with clear paragraph breaks. The essay exceeds 
the minimum word requirement and stays within limits. The conclusion 
effectively summarizes the main arguments. As a result, the overall quality 
is excellent. Therefore, this essay demonstrates strong academic writing 
skills and deep understanding of the subject matter.
"""

lorem_paragraph = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut "
    "labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
    "laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in "
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat "
    "non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
)

# Long enough to clear default min_words=250 and include multiple paragraphs.
lorem_essay = (lorem_paragraph + "\n\n") * 8

question = "Discuss the importance of education in modern society."

print("=" * 70)
print("COMPONENT-BASED ESSAY SCORING TEST")
print("=" * 70)
print(f"\nQuestion: {question}")
print(f"Rubric: Content (50) + Grammar (30) + Formatting (20) = 100 total\n")

print("-" * 70)
print("WEAK ESSAY")
print("-" * 70)
score, g, r, o, highlighted = total_score(
    question, weak_essay, debug=True
)
print(f"\nFinal Score: {score}/100")
print(f"  Content: {r}/50 | Grammar: {g}/30 | Formatting: {o}/20")
print()

print("-" * 70)
print("AVERAGE ESSAY")
print("-" * 70)
score, g, r, o, highlighted = total_score(
    question, average_essay, debug=True
)
print(f"\nFinal Score: {score}/100")
print(f"  Content: {r}/50 | Grammar: {g}/30 | Formatting: {o}/20")
print()

print("-" * 70)
print("STRONG ESSAY")
print("-" * 70)
score, g, r, o, highlighted = total_score(
    question, strong_essay, debug=True
)
print(f"\nFinal Score: {score}/100")
print(f"  Content: {r}/50 | Grammar: {g}/30 | Formatting: {o}/20")
print()

print("-" * 70)
print("LOREM IPSUM (PLACEHOLDER) ESSAY")
print("-" * 70)
score, g, r, o, highlighted = total_score(
    question, lorem_essay, debug=True
)
print(f"\nFinal Score: {score}/100")
print(f"  Content: {r}/50 | Grammar: {g}/30 | Formatting: {o}/20")
print()

print("=" * 70)
print("SCORE SPREAD VERIFICATION")
print("=" * 70)
print("\nIf scores are well-spread (not all in 60–64 range), component-based")
print("approach is working correctly. ML now drives content (0–50), not final score.")
print("=" * 70)
