import re
from django.shortcuts import render
from django.utils.safestring import mark_safe
from .ml.scoring import total_score


def home(request):
    score = None
    g = r = o = None
    error = None
    highlighted_essay = None

    min_words = 100
    max_words = 1000
    content_points = 50
    grammar_points = 30
    formatting_points = 20

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        essay = request.POST.get("essay", "").strip()

        try:
            min_words = int(request.POST.get("min_words") or 100)
            max_words = int(request.POST.get("max_words") or 1000)
            content_points = int(request.POST.get("content_points") or 50)
            grammar_points = int(request.POST.get("grammar_points") or 30)
            formatting_points = int(request.POST.get("formatting_points") or 20)
        except ValueError:
            error = "Please enter valid numbers."

        word_count = len(re.findall(r'\b\w+\b', essay))

        if min_words < 100:
            error = "Minimum words cannot be lower than 100."
        elif max_words < min_words:
            error = "Maximum words must be greater than or equal to minimum words."
        elif (content_points + grammar_points + formatting_points) != 100:
            error = "Scoring weights must total exactly 100."
        elif not question or not essay:
            error = "Please enter both the essay question and the essay."
        elif word_count < min_words:
            error = f"Essay must be at least {min_words} words."
        elif word_count > max_words:
            error = f"Essay must not exceed {max_words} words."
        else:
            score, g, r, o, highlighted_essay = total_score(
                question=question,
                essay=essay,
                content_points=content_points,
                grammar_points=grammar_points,
                formatting_points=formatting_points,
                min_words=min_words,
                max_words=max_words,
            )

    return render(request, 'scorer/home.html', {
        'score': score,
        'grammar': g,
        'relevance': r,
        'other': o,
        'error': error,
        'highlighted_essay': mark_safe(highlighted_essay) if highlighted_essay else None,
        'min_words': min_words,
        'max_words': max_words,
        'content_points': content_points,
        'grammar_points': grammar_points,
        'formatting_points': formatting_points,
    })